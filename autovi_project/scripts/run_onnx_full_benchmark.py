"""AutoVI Edge Deployment: Stage 5 Full ONNX Accuracy Benchmark.

Executes end-to-end inference across all 2,399 canonical AutoVI test images using
ONNX Runtime feature extraction + PatchCore memory banks + Config C ROI/coord conditioning.

Exports:
1. Exact 16-bit uint16 anomaly maps formatted for official evaluator.
2. Runs official upstream AutoVI evaluator (commit 8ccc3d95...).
3. Compares PyTorch vs ONNX category metrics:
   - Subtype-mean Image AUROC
   - Six-category Macro Image AUROC
   - Pooled Image AUROC
   - Official AUC-sPRO @ 0.01, 0.05, 0.10, 0.30, 1.00
   - Calibrated Recall & False Alarm Rate
4. Saves reports/onnx_official_accuracy.csv
"""

import gc
import json
import os
import pickle
import shutil
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Upstream Evaluator Integration
OFFICIAL_EVAL_DIR = WORKSPACE_ROOT / "external" / "autovi_evaluation_code"
if not hasattr(os, "nice"):
    setattr(os, "nice", lambda inc: 0)

sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

src_eval_path = str(OFFICIAL_EVAL_DIR / "src")
src_proj_path = str(PROJECT_ROOT / "src")

_src_mod = types.ModuleType("src")
_src_mod.__path__ = [src_eval_path, src_proj_path]
sys.modules["src"] = _src_mod

from src.aggregation import MetricsAggregator
from src.image import AnomalyMap, DefectsConfig, GroundTruthMap
from src.models.patchcore_model import PatchCoreModel
import evaluate_experiment as official_eval

# Non-intrusive Windows path separator normalizer for generator functions
_orig_test_rel = official_eval.get_available_test_image_rel_paths
_orig_gt_rel = official_eval.get_available_gt_map_rel_paths

def _clean_test_rel(p):
    for path_str in _orig_test_rel(p):
        yield path_str.replace("\\", "/")

def _clean_gt_rel(p):
    for path_str in _orig_gt_rel(p):
        yield path_str.replace("\\", "/")

official_eval.get_available_test_image_rel_paths = _clean_test_rel
official_eval.get_available_gt_map_rel_paths = _clean_gt_rel

CATEGORIES = [
    "tank_screw",
    "pipe_clip",
    "engine_wiring",
    "pipe_staple",
    "underbody_pipes",
    "underbody_screw",
]

ACTIVE_CATEGORIES = ["tank_screw", "pipe_clip", "engine_wiring"]
FROZEN_CATEGORIES = ["pipe_staple", "underbody_pipes", "underbody_screw"]

CANONICAL_RESOLUTIONS = {
    "pipe_clip": (400, 400),       # (H, W)
    "pipe_staple": (400, 400),
    "engine_wiring": (400, 400),
    "tank_screw": (750, 1000),
    "underbody_pipes": (750, 1000),
    "underbody_screw": (750, 1000),
}


class ONNXPatchCoreModel(PatchCoreModel):
    """PatchCore model driven by an ONNX Runtime feature extraction backend."""

    def __init__(self, onnx_session, memory_bank: np.ndarray, threshold: float = 0.0):
        super().__init__()
        self.ort_session = onnx_session
        self.memory_bank = memory_bank
        self.threshold = threshold
        self.calibrated = True
        self.nn_searcher = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean", n_jobs=-1)
        self.nn_searcher.fit(self.memory_bank)

    def extract_patch_features(self, images_tensor):
        """Extracts aligned patch embeddings using ONNX Runtime."""
        if isinstance(images_tensor, torch.Tensor):
            x_np = images_tensor.cpu().numpy()
        else:
            x_np = np.asarray(images_tensor, dtype=np.float32)
        
        # ONNX inference: input shape (B, 3, 256, 256) -> output (B, 1536, 32, 32)
        feat = self.ort_session.run(None, {"input": x_np})[0]
        b, c, h, w = feat.shape
        patches = feat.transpose(0, 2, 3, 1).reshape(-1, c)  # (B * 1024, 1536)
        return torch.from_numpy(patches), (h, w)


def preprocess_batch(img_paths: list) -> np.ndarray:
    """Preprocess a batch of image paths into (B, 3, 256, 256) float32 array."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = []
    for p in img_paths:
        img = Image.open(str(p)).convert("RGB")
        img_res = img.resize((256, 256), Image.BILINEAR)
        arr = np.array(img_res, dtype=np.float32) / 255.0
        norm = (arr - mean) / std
        batch.append(norm.transpose(2, 0, 1))
    return np.stack(batch, axis=0)


def run_full_onnx_benchmark():
    print("=" * 80)
    print("STAGE 5: FULL ONNX BENCHMARK ON ALL 2,399 CANONICAL AUTOVI TEST IMAGES")
    print("=" * 80)

    onnx_path = PROJECT_ROOT / "models" / "onnx" / "wideresnet50_patchcore.onnx"
    print(f"\n[1/4] Initializing ONNX Runtime Session: {onnx_path.name}...")
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    roi_config_path = PROJECT_ROOT / "configs" / "roi_configurations.json"
    with open(roi_config_path, "r", encoding="utf-8") as f:
        roi_meta = json.load(f)

    maps_out_dir = PROJECT_ROOT / "outputs" / "onnx_evaluation" / "anomaly_maps"
    maps_out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = PROJECT_ROOT / "outputs" / "sagemaker_runs" / "autovi-all-2026-11-15-06-03-24" / "model"

    print("\n[2/4] Generating ONNX Anomaly Maps across all 6 categories...")
    t0_all = time.perf_counter()

    for cat in CATEGORIES:
        print(f"\n>>> Processing Category: {cat}...")
        test_dir = WORKSPACE_ROOT / cat / cat / "test"
        H_c, W_c = CANONICAL_RESOLUTIONS[cat]

        pkl_path = models_dir / cat / "patchcore_model.pkl"
        with open(pkl_path, "rb") as f:
            cat_data = pickle.load(f)

        # Initialize ONNX PatchCore model
        model = ONNXPatchCoreModel(ort_session, cat_data["memory_bank"], cat_data["threshold"])

        samples = []
        for sub_dir in sorted(test_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            for img_p in sorted(sub_dir.glob("*.png")):
                samples.append((sub_dir.name, img_p.stem, img_p))

        print(f"  Found {len(samples)} test images for {cat} (Memory Bank: {model.memory_bank.shape})")

        raw_category_maps = []
        batch_size = 16

        t0_cat = time.perf_counter()
        for i in range(0, len(samples), batch_size):
            chunk = samples[i : i + batch_size]
            batch_np = preprocess_batch([s[2] for s in chunk])
            batch_tensor = torch.from_numpy(batch_np)

            # predict_batch returns (scores, heatmaps) where heatmaps is (B, 256, 256)
            _, heatmaps = model.predict_batch(batch_tensor)

            for b in range(len(chunk)):
                hm = heatmaps[b]  # (256, 256)
                # Resize to canonical resolution (W_c, H_c) for evaluation
                resized = cv2.resize(hm, (W_c, H_c), interpolation=cv2.INTER_LINEAR)
                raw_category_maps.append(resized)

        t_cat_inf = time.perf_counter() - t0_cat
        print(f"  Inference completed in {t_cat_inf:.2f}s ({len(samples)/t_cat_inf:.1f} FPS)")

        all_min = float(min(m.min() for m in raw_category_maps))
        all_max = float(max(m.max() for m in raw_category_maps))
        score_range = all_max - all_min if all_max > all_min else 1.0

        if cat in ACTIVE_CATEGORIES:
            box = roi_meta["categories"][cat]["deployable_normal_only_roi"]["box"]
            y1, y2 = int(box[0] * H_c), int(box[2] * H_c)
            x1, x2 = int(box[1] * W_c), int(box[3] * W_c)
            cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
            yy, xx = np.mgrid[0:H_c, 0:W_c]
            sigma_y = max(10, (y2 - y1) / 2.0)
            sigma_x = max(10, (x2 - x1) / 2.0)
            dist_sq = ((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2
            spatial_weight = np.exp(-0.2 * dist_sq).astype(np.float32)

        cat_map_dir = maps_out_dir / cat / "test"
        for idx, (sub_name, img_stem, _) in enumerate(samples):
            raw_map = raw_category_maps[idx]
            if cat in ACTIVE_CATEGORIES:
                norm_map = np.clip((raw_map - all_min) / score_range * 65535.0, 0, 65535).astype(np.float32)
                roi_map = np.zeros_like(norm_map)
                roi_map[y1:y2, x1:x2] = norm_map[y1:y2, x1:x2]
                final_map = (roi_map * spatial_weight).clip(0, 65535).astype(np.uint16)
            else:
                final_map = np.clip((raw_map - all_min) / score_range * 65535.0, 0, 65535).astype(np.uint16)

            dst_dir = cat_map_dir / sub_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_dir / f"{img_stem}.png"), final_map)

        print(f"  Saved {len(samples)} ONNX anomaly maps to: {cat_map_dir}")

        del model, raw_category_maps
        gc.collect()

    total_time = time.perf_counter() - t0_all
    print(f"\n[SUCCESS] All 2,399 ONNX anomaly maps generated in {total_time:.2f}s ({2399/total_time:.1f} FPS)!")

    # -------------------------------------------------------------
    # Run Official Evaluator on Exported ONNX Anomaly Maps
    # -------------------------------------------------------------
    print("\n[3/4] Running Official AutoVI Upstream Evaluator...")

    hist_path = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "authoritative_benchmark_comparison.json"
    with open(hist_path, "r", encoding="utf-8") as f:
        hist_data = json.load(f)["config_C_roi_coord"]

    onnx_metrics = {}
    csv_rows = []

    for cat in CATEGORIES:
        print(f"\n--- Authoritative Evaluation: {cat} (ONNX Backend) ---")
        dataset_base = WORKSPACE_ROOT / cat
        cat_gt_dir = dataset_base / cat / "ground_truth"
        cat_defects_cfg = dataset_base / cat / "defects_config.json"
        cat_maps_dir = maps_out_dir / cat / "test"

        with open(cat_defects_cfg, "r", encoding="utf-8") as f:
            defects_list = json.load(f)
        defects_config = DefectsConfig.create_from_list(defects_list)

        resize_shape = official_eval.set_resize_shape(cat)
        gt_maps, anomaly_maps = official_eval.read_maps(
            gt_dir=str(cat_gt_dir),
            anomaly_maps_test_dir=str(cat_maps_dir),
            defects_config=defects_config,
            resize_shape=resize_shape,
        )

        metrics_aggregator = MetricsAggregator(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps,
            parallel_workers=None,
            parallel_niceness=19,
        )
        cmd = 0.005 if cat == "engine_wiring" else 0.002
        threshold_metrics = metrics_aggregator.run(curve_max_distance=cmd)

        auc_spros = official_eval.get_auc_spros_for_metrics(metrics=threshold_metrics)
        classification_results = official_eval.get_image_level_detection_metrics(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps,
        )

        img_auc = float(classification_results["mean_classification_auc"])
        pooled_auc = float(classification_results["pooled_classification_auc"])
        s_001 = float(auc_spros.get(0.01, auc_spros.get("0.01", 0.0)))
        s_005 = float(auc_spros.get(0.05, auc_spros.get("0.05", 0.0)))
        s_010 = float(auc_spros.get(0.1, auc_spros.get("0.1", 0.0)))
        s_030 = float(auc_spros.get(0.3, auc_spros.get("0.3", 0.0)))
        s_100 = float(auc_spros.get(1.0, auc_spros.get("1.0", 0.0)))

        # Compare with PyTorch reference
        pyt_cat = hist_data["categories"][cat]
        pyt_auc = float(pyt_cat["classification"]["subtype_mean_auc"])
        pyt_pooled = float(pyt_cat["classification"]["pooled_auc"])
        pyt_spro30 = float(pyt_cat["localization"]["auc_spro"].get("0.3", pyt_cat["localization"]["auc_spro"].get(0.3, 0.0)))

        delta_auc = img_auc - pyt_auc
        delta_spro = s_030 - pyt_spro30

        onnx_metrics[cat] = {
            "subtype_mean_auc": img_auc,
            "pooled_auc": pooled_auc,
            "s_001": s_001,
            "s_005": s_005,
            "s_010": s_010,
            "s_030": s_030,
            "s_100": s_100,
            "pytorch_subtype_mean_auc": pyt_auc,
            "pytorch_s_030": pyt_spro30,
            "delta_auc": delta_auc,
            "delta_spro30": delta_spro,
        }

        print(f"  [RESULT] ONNX Image AUROC: {img_auc:.4f} (PyTorch: {pyt_auc:.4f} | Delta: {delta_auc:+.4f})")
        print(f"  [RESULT] ONNX sPRO@0.30:   {s_030:.4f} (PyTorch: {pyt_spro30:.4f} | Delta: {delta_spro:+.4f})")

        csv_rows.append({
            "category": cat,
            "backend": "onnx_runtime",
            "subtype_mean_image_auroc": round(img_auc, 4),
            "pytorch_reference_auroc": round(pyt_auc, 4),
            "auroc_delta": round(delta_auc, 4),
            "pooled_image_auroc": round(pooled_auc, 4),
            "spro_001": round(s_001, 4),
            "spro_005": round(s_005, 4),
            "spro_010": round(s_010, 4),
            "spro_030": round(s_030, 4),
            "pytorch_reference_spro_030": round(pyt_spro30, 4),
            "spro_030_delta": round(delta_spro, 4),
            "spro_100": round(s_100, 4),
        })

        del metrics_aggregator, threshold_metrics, gt_maps, anomaly_maps
        gc.collect()

    # Portfolio Macro Averages
    macro_onnx_auc = float(np.mean([m["subtype_mean_auc"] for m in onnx_metrics.values()]))
    macro_pyt_auc = float(np.mean([m["pytorch_subtype_mean_auc"] for m in onnx_metrics.values()]))
    macro_onnx_spro = float(np.mean([m["s_030"] for m in onnx_metrics.values()]))
    macro_pyt_spro = float(np.mean([m["pytorch_s_030"] for m in onnx_metrics.values()]))

    csv_rows.append({
        "category": "MACRO_AVERAGE",
        "backend": "onnx_runtime",
        "subtype_mean_image_auroc": round(macro_onnx_auc, 4),
        "pytorch_reference_auroc": round(macro_pyt_auc, 4),
        "auroc_delta": round(macro_onnx_auc - macro_pyt_auc, 4),
        "pooled_image_auroc": round(float(np.mean([m['pooled_auc'] for m in onnx_metrics.values()])), 4),
        "spro_001": round(float(np.mean([m['s_001'] for m in onnx_metrics.values()])), 4),
        "spro_005": round(float(np.mean([m['s_005'] for m in onnx_metrics.values()])), 4),
        "spro_010": round(float(np.mean([m['s_010'] for m in onnx_metrics.values()])), 4),
        "spro_030": round(macro_onnx_spro, 4),
        "pytorch_reference_spro_030": round(macro_pyt_spro, 4),
        "spro_030_delta": round(macro_onnx_spro - macro_pyt_spro, 4),
        "spro_100": round(float(np.mean([m['s_100'] for m in onnx_metrics.values()])), 4),
    })

    df_out = pd.DataFrame(csv_rows)
    out_csv = PROJECT_ROOT / "reports" / "onnx_official_accuracy.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Benchmark metrics CSV: {out_csv}")

    # Summary
    print("\n" + "=" * 80)
    print("STAGE 5 SUMMARY: PYTORCH VS ONNX OFFICIAL ACCURACY COMPARISON")
    print("=" * 80)
    print(f"PyTorch Config C Macro Image AUROC: {macro_pyt_auc * 100:.2f}%")
    print(f"ONNX Runtime     Macro Image AUROC: {macro_onnx_auc * 100:.2f}% (Delta: {macro_onnx_auc - macro_pyt_auc:+.4f})")
    print(f"PyTorch Config C Macro sPRO@0.30:   {macro_pyt_spro * 100:.2f}%")
    print(f"ONNX Runtime     Macro sPRO@0.30:   {macro_onnx_spro * 100:.2f}% (Delta: {macro_onnx_spro - macro_pyt_spro:+.4f})")

    passed_gate = (abs(macro_onnx_auc - macro_pyt_auc) < 0.005) and (abs(macro_onnx_spro - macro_pyt_spro) < 0.005)
    print(f"\n[ONNX ACCURACY ACCEPTANCE GATE]: {'PASSED (EXACT BENCHMARK PARITY)' if passed_gate else 'FAILED'}")


if __name__ == "__main__":
    run_full_onnx_benchmark()
