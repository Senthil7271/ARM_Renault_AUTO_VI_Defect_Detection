#!/usr/bin/env python3
"""AutoVI Edge: Dedicated ONNX Runtime CPU Test & Evaluation Script.

Runs WideResNet-50-2 PatchCore on ARM Cortex-A76 CPU using ONNX Runtime.
Uses Config C (Deployable Normal ROI + Spatial Coordinate Conditioning + Gaussian Blur).

Usage:
  python3 evaluate_onnx_cpu.py --category pipe_clip
  python3 evaluate_onnx_cpu.py --category pipe_staple
  python3 evaluate_onnx_cpu.py --category engine_wiring
  python3 evaluate_onnx_cpu.py --category all
"""

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import sys
import time

# Suppress verbose ONNX schema warnings on Debian
os.environ["ORT_LOG_LEVEL"] = "3"
os.environ["ONNX_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent

# Auto-detect Model Directory
candidate_model_dirs = [
    ROOT / "models",
    Path("/sdcard/autovi-mobile-package/models"),
    ROOT.parent / "models",
    Path("/root/models"),
]
MODEL_DIR = next((p for p in candidate_model_dirs if p.exists()), ROOT / "models")
ONNX_PATH = MODEL_DIR / "wideresnet50_patchcore.onnx"
MB_DIR = MODEL_DIR / "memory_banks"

# Auto-detect Config File
candidate_configs = [
    ROOT / "configs" / "roi_configurations.json",
    Path("/sdcard/autovi-mobile-package/configs/roi_configurations.json"),
    ROOT.parent / "configs" / "roi_configurations.json",
]
CONFIG_FILE = next((p for p in candidate_configs if p.exists()), ROOT / "configs" / "roi_configurations.json")

# Auto-detect Stream Directory & Ground Truth
candidate_stream_dirs = [
    ROOT / "test_stream",
    ROOT / "test_stream" / "test_stream",
    Path("/sdcard/test_stream/test_stream"),
    Path("/sdcard/test_stream"),
    Path("/sdcard/Download/test_stream"),
    Path("/mnt/sdcard/test_stream/test_stream"),
    Path("/mnt/sdcard/test_stream"),
]
STREAM_DIR = next((p for p in candidate_stream_dirs if (p / "ground_truth.csv").exists()), ROOT / "test_stream")
GT_FILE = STREAM_DIR / "ground_truth.csv"

# Recommended Calibrated Operating Thresholds
CALIBRATED_THRESHOLDS = {
    "pipe_clip": 16.75,
    "pipe_staple": 17.28,
    "engine_wiring": 17.85,
    "tank_screw": 16.50,
    "underbody_pipes": 21.34,
    "underbody_screw": 14.44,
}

# ImageNet normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ONNXAnomalyEvaluator:
    def __init__(self, onnx_model_path: Path, num_threads: int = 4):
        if not onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_model_path}")

        print(f"Loading ONNX Model: {onnx_model_path} ({onnx_model_path.stat().st_size / 1e6:.1f} MB)...")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX Runtime Initialized on CPU ({num_threads} threads).")

        # Load ROI configurations
        self.roi_configs = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self.roi_configs = json.load(f).get("categories", {})

        # Cache memory banks
        self.banks = {}
        self.active_cat = None

    def set_category(self, category: str, threshold_override: float = None):
        if category in self.banks and threshold_override is None:
            self.active_cat = category
            return

        pkl_path = MB_DIR / f"{category}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Memory bank not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        mb = np.asarray(data["memory_bank"], dtype=np.float32)
        mb_sq = np.einsum("ij,ij->i", mb, mb)
        
        if threshold_override is not None:
            threshold = float(threshold_override)
        elif category in CALIBRATED_THRESHOLDS:
            threshold = float(CALIBRATED_THRESHOLDS[category])
        else:
            threshold = float(data.get("threshold", 17.50))

        # Setup ROI and Config C Coordinate Conditioning
        roi_meta = None
        if category in self.roi_configs:
            box = self.roi_configs[category]["deployable_normal_only_roi"]["box"]
            gy1 = max(0, int(np.floor(box[0] * 32)))
            gx1 = max(0, int(np.floor(box[1] * 32)))
            gy2 = min(32, int(np.ceil(box[2] * 32)))
            gx2 = min(32, int(np.ceil(box[3] * 32)))

            # Coordinate falloff centered on component envelope
            cy, cx = (gy1 + gy2) / 2.0, (gx1 + gx2) / 2.0
            sy = max(1.0, (gy2 - gy1) / 2.0)
            sx = max(1.0, (gx2 - gx1) / 2.0)
            yy, xx = np.mgrid[0:32, 0:32]
            dist_sq = ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2
            spatial_weight = np.exp(-0.2 * dist_sq).astype(np.float32)

            roi_meta = {
                "gy1": gy1, "gy2": gy2, "gx1": gx1, "gx2": gx2,
                "spatial_weight": spatial_weight,
            }

        self.banks[category] = {
            "mb": mb,
            "mb_sq": mb_sq,
            "threshold": threshold,
            "roi": roi_meta,
        }
        self.active_cat = category
        print(f"Loaded memory bank for '{category}': shape {mb.shape}, tau={threshold:.4f} (Calibrated Operating Point)")

    def inspect(self, img_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()

        # 1. Preprocess
        t_prep_start = time.perf_counter()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        norm = (img_res - MEAN) / STD
        tensor = norm.transpose(2, 0, 1)[None, ...].astype(np.float32)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # 2. ONNX Backbone Inference
        t_nn_inf_start = time.perf_counter()
        raw_out = self.session.run(None, {self.input_name: tensor})[0]
        # raw_out is [1, 1536, 32, 32] -> transpose to [32, 32, 1536]
        patches_grid = raw_out[0].transpose(1, 2, 0)
        t_backbone_ms = (time.perf_counter() - t_nn_inf_start) * 1000.0

        # 3. Nearest-Neighbor Matching
        t_match_start = time.perf_counter()
        cat_info = self.banks[self.active_cat]
        mb = cat_info["mb"]
        mb_sq = cat_info["mb_sq"]
        roi = cat_info["roi"]
        tau = cat_info["threshold"]

        if roi is not None:
            gy1, gy2, gx1, gx2 = roi["gy1"], roi["gy2"], roi["gx1"], roi["gx2"]
            roi_patches = patches_grid[gy1:gy2, gx1:gx2].reshape(-1, 1536)

            p_sq = np.sum(roi_patches ** 2, axis=1, keepdims=True)
            dots = np.dot(roi_patches, mb.T)
            d2 = np.maximum(p_sq + mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1))

            roi_dists = min_d.reshape(gy2 - gy1, gx2 - gx1)
            # Apply Config C coordinate weighting
            sw = roi["spatial_weight"][gy1:gy2, gx1:gx2]
            weighted_roi = roi_dists * sw

            # 2D Gaussian smoothing (sigma=1.2)
            smoothed_roi = cv2.GaussianBlur(weighted_roi, (5, 5), sigmaX=1.2)
            score = float(np.max(smoothed_roi))
        else:
            full_patches = patches_grid.reshape(-1, 1536)
            p_sq = np.sum(full_patches ** 2, axis=1, keepdims=True)
            dots = np.dot(full_patches, mb.T)
            d2 = np.maximum(p_sq + mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1))
            dist_2d = min_d.reshape(32, 32)
            smoothed_2d = cv2.GaussianBlur(dist_2d, (5, 5), sigmaX=1.2)
            score = float(np.max(smoothed_2d))

        t_match_ms = (time.perf_counter() - t_match_start) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "score": score,
            "threshold": tau,
            "is_reject": score >= tau,
            "latency_ms": round(total_ms, 1),
            "breakdown": {
                "prep": round(t_prep_ms, 1),
                "onnx": round(t_backbone_ms, 1),
                "nn": round(t_match_ms, 1),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="AutoVI Dedicated ONNX CPU Evaluator")
    parser.add_argument("--category", default="pipe_clip", help="Category name or 'all'")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold override (defaults to calibrated tau)")
    parser.add_argument("--csv", default="", help="Output CSV filename (auto-named if omitted)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for ONNX Runtime")
    args = parser.parse_args()

    if not GT_FILE.exists():
        sys.exit(f"[ERROR] Ground truth file not found at: {GT_FILE}\nPlease ensure test_stream directory with ground_truth.csv exists.")

    # Load items
    items = []
    with open(GT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if args.category == "all" or r["category"] == args.category:
                items.append(r)

    if not items:
        sys.exit(f"[ERROR] No images found for category '{args.category}'.")

    if args.limit > 0:
        items = items[: args.limit]

    out_csv = args.csv if args.csv else f"{args.category}_onnx_eval.csv"

    print("=" * 110)
    print(" AUTOVI INDUSTRIAL EVALUATION AUDIT (ONNX RUNTIME CPU)")
    print(f" Target: {args.category} | Model: {ONNX_PATH.name} | Threads: {args.threads}")
    print(f" Stream Root: {STREAM_DIR} | Ground Truth: {GT_FILE}")
    print(f" Images: {len(items)} (Good: {sum(1 for x in items if x['ground_truth'] == 'GOOD')}, Defect: {sum(1 for x in items if x['ground_truth'] == 'DEFECT')})")
    print("=" * 110)

    evaluator = ONNXAnomalyEvaluator(ONNX_PATH, num_threads=args.threads)

    # Setup CSV output
    csv_file = open(out_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "category", "filename", "ground_truth", "defect_type",
        "anomaly_score", "threshold", "verdict", "evaluation_result", "latency_ms"
    ])

    print(f"\n{'FRAME':<6} | {'CATEGORY':<14} | {'FILE':<22} | {'GT':<7} | {'SCORE':<7} | {'TAU':<6} | {'VERDICT':<10} | {'RESULT':<12} | {'LATENCY':<9}")
    print("-" * 110)

    cat_stats = {}
    current_cat = None

    for idx, item in enumerate(items, start=1):
        cat = item["category"]
        rel_path = item["filename"]
        gt = item["ground_truth"]
        defect_type = item.get("defect_type", "")

        if cat not in cat_stats:
            cat_stats[cat] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0, "time_ms": 0.0, "good": 0, "defect": 0}
        cat_stats[cat]["total"] += 1
        if gt == "DEFECT":
            cat_stats[cat]["defect"] += 1
        else:
            cat_stats[cat]["good"] += 1

        img_p = STREAM_DIR / cat / rel_path
        if not img_p.exists():
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            continue

        if cat != current_cat:
            evaluator.set_category(cat, threshold_override=args.threshold)
            current_cat = cat

        res = evaluator.inspect(img)
        score = res["score"]
        tau = res["threshold"]
        lat = res["latency_ms"]
        cat_stats[cat]["time_ms"] += lat

        is_reject = res["is_reject"]
        verdict = "REJECT" if is_reject else "PASS"

        if gt == "DEFECT":
            if is_reject:
                cat_stats[cat]["tp"] += 1
                res_str = "\033[92mMATCH [TP]\033[0m"
                eval_res = "CORRECT (TP)"
            else:
                cat_stats[cat]["fn"] += 1
                res_str = "\033[91mESCAPE [FN]\033[0m"
                eval_res = "ESCAPE (FN)"
        else:
            if not is_reject:
                cat_stats[cat]["tn"] += 1
                res_str = "\033[92mMATCH [TN]\033[0m"
                eval_res = "CORRECT (TN)"
            else:
                cat_stats[cat]["fp"] += 1
                res_str = "\033[91mFALSE [FP]\033[0m"
                eval_res = "FALSE_ALARM (FP)"

        v_str = "\033[91mREJECT   \033[0m" if is_reject else "\033[92mPASS [OK]\033[0m"
        short_file = rel_path if len(rel_path) <= 22 else "..." + rel_path[-19:]

        print(f"#{idx:<5} | {cat:<14} | {short_file:<22} | {gt:<7} | {score:<7.3f} | {tau:<6.2f} | {v_str} | {res_str} | {lat:6.1f} ms", flush=True)

        csv_writer.writerow([
            idx, cat, rel_path, gt, defect_type,
            round(score, 4), round(tau, 4), verdict, eval_res, lat
        ])

    csv_file.close()

    # Print Summary Audit Table
    print("\n" + "=" * 115)
    print(" AUTOVI INDUSTRIAL EVALUATION SUMMARY AUDIT (ONNX RUNTIME CPU)")
    print("=" * 115)
    print(f"{'CATEGORY':<18} | {'TOTAL':<6} | {'GOOD':<5} | {'DEFECT':<6} | {'TP':<4} | {'TN':<4} | {'FP':<4} | {'FN':<4} | {'ACCURACY':<9} | {'RECALL':<8} | {'AVG LAT':<8}")
    print("-" * 115)

    tot_tp, tot_tn, tot_fp, tot_fn, tot_all, tot_time = 0, 0, 0, 0, 0, 0.0
    for c_name, st in sorted(cat_stats.items()):
        acc = ((st["tp"] + st["tn"]) / st["total"]) * 100.0 if st["total"] > 0 else 0.0
        rec = (st["tp"] / st["defect"]) * 100.0 if st["defect"] > 0 else 0.0
        avg_l = st["time_ms"] / st["total"] if st["total"] > 0 else 0.0
        print(f"{c_name:<18} | {st['total']:<6} | {st['good']:<5} | {st['defect']:<6} | {st['tp']:<4} | {st['tn']:<4} | {st['fp']:<4} | {st['fn']:<4} | {acc:8.2f}% | {rec:7.2f}% | {avg_l:6.1f} ms")
        tot_tp += st["tp"]
        tot_tn += st["tn"]
        tot_fp += st["fp"]
        tot_fn += st["fn"]
        tot_all += st["total"]
        tot_time += st["time_ms"]

    print("-" * 115)
    o_acc = ((tot_tp + tot_tn) / tot_all) * 100.0 if tot_all > 0 else 0.0
    o_rec = (tot_tp / (tot_tp + tot_fn)) * 100.0 if (tot_tp + tot_fn) > 0 else 0.0
    o_lat = tot_time / tot_all if tot_all > 0 else 0.0
    print(f"{'OVERALL':<18} | {tot_all:<6} | {tot_all - (tot_tp + tot_fn):<5} | {tot_tp + tot_fn:<6} | {tot_tp:<4} | {tot_tn:<4} | {tot_fp:<4} | {tot_fn:<4} | {o_acc:8.2f}% | {o_rec:7.2f}% | {o_lat:6.1f} ms")
    print("=" * 115)
    print(f" Detailed Audit CSV Saved To: {out_csv}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
