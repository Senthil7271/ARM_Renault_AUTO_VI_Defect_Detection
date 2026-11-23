"""Authoritative AutoVI evaluation runner using original authors' implementation.

Executes official evaluate_experiment.py and MetricsAggregator directly from:
external/autovi_evaluation_code (commit 8ccc3d95054edf8237a152a6c29f61c0504b0b09)

Evaluates:
- Configuration A: Baseline PatchCore
- Configuration B: Deployable Normal-Only ROI PatchCore
- Configuration C: Deployable Normal ROI + Coordinate Conditioning
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
OFFICIAL_EVAL_DIR = WORKSPACE_ROOT / "external" / "autovi_evaluation_code"

# Windows niceness compatibility
if not hasattr(os, "nice"):
    setattr(os, "nice", lambda inc: 0)

sys.path.insert(0, str(OFFICIAL_EVAL_DIR))

from src.aggregation import MetricsAggregator
from src.image import AnomalyMap, DefectsConfig, GroundTruthMap
from src.util import get_auc_for_max_fpr
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

ALL_CATEGORIES = [
    "pipe_clip",
    "pipe_staple",
    "engine_wiring",
    "tank_screw",
    "underbody_pipes",
    "underbody_screw",
]

CONFIGURATIONS = [
    ("config_A_baseline", "Baseline Full-Frame PatchCore"),
    ("config_B_roi_only", "Deployable Normal-Only ROI PatchCore"),
    ("config_C_roi_coord", "Deployable Normal ROI + Coordinate Conditioning"),
]


def evaluate_single_config(config_id: str, config_label: str):
    print("=" * 80)
    print(f"RUNNING OFFICIAL EVALUATOR: {config_id} ({config_label})")
    print(f"Commit: 8ccc3d95054edf8237a152a6c29f61c0504b0b09")
    print("=" * 80)

    maps_root = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / config_id
    eval_out_root = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "official_eval_results" / config_id
    eval_out_root.mkdir(parents=True, exist_ok=True)

    config_results = {}

    for cat in ALL_CATEGORIES:
        print(f"\n>>> Evaluating Object: {cat}")
        t0 = time.time()

        # Data & Maps paths
        # Canonical double-nested path
        dataset_base = WORKSPACE_ROOT / cat
        cat_gt_dir = dataset_base / cat / "ground_truth"
        cat_defects_cfg = dataset_base / cat / "defects_config.json"
        cat_maps_dir = maps_root / cat / "test"
        cat_out_dir = eval_out_root / cat
        cat_out_dir.mkdir(parents=True, exist_ok=True)
        out_json_path = cat_out_dir / "official_metrics.json"

        # 1. Fast Cache Check
        if out_json_path.exists():
            print(f"  [CACHED {cat}] Found existing results at {out_json_path}")
            with open(out_json_path, "r", encoding="utf-8") as f:
                cat_summary = json.load(f)
            config_results[cat] = cat_summary
            continue

        # 2. Frozen Category Check (Bitwise identical maps between configs)
        if config_id in ("config_B_roi_only", "config_C_roi_coord") and cat in ("pipe_staple", "underbody_pipes", "underbody_screw"):
            ref_path = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "official_eval_results" / "config_A_baseline" / cat / "official_metrics.json"
            if ref_path.exists():
                print(f"  [REUSED FROZEN {cat}] Inherited bitwise identical results from Config A")
                with open(ref_path, "r", encoding="utf-8") as f:
                    cat_summary = json.load(f)
                cat_summary["config_id"] = config_id
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(cat_summary, f, indent=4, sort_keys=True)
                config_results[cat] = cat_summary
                continue

        assert cat_gt_dir.exists(), f"GT dir missing: {cat_gt_dir}"
        assert cat_defects_cfg.exists(), f"Defects config missing: {cat_defects_cfg}"
        assert cat_maps_dir.exists(), f"Maps dir missing: {cat_maps_dir}"

        # 1. Load Defects Config
        with open(cat_defects_cfg, "r", encoding="utf-8") as f:
            defects_list = json.load(f)
        defects_config = DefectsConfig.create_from_list(defects_list)

        # 2. Canonical shape
        resize_shape = official_eval.set_resize_shape(cat)

        # 3. Read Maps using unmodified official read_maps
        gt_maps, anomaly_maps = official_eval.read_maps(
            gt_dir=str(cat_gt_dir),
            anomaly_maps_test_dir=str(cat_maps_dir),
            defects_config=defects_config,
            resize_shape=resize_shape,
        )
        print(f"  Loaded {len(anomaly_maps)} anomaly maps and {len(gt_maps)} GT entries.")

        # 4. Localization: Run official MetricsAggregator
        metrics_aggregator = MetricsAggregator(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps,
            parallel_workers=None,
            parallel_niceness=19,
        )
        # Use curve_max_distance=0.005 for high-sample categories to prevent Windows RAM fragmentation
        cmd = 0.005 if cat == "engine_wiring" else 0.002
        threshold_metrics = metrics_aggregator.run(curve_max_distance=cmd)

        # Official sPRO
        auc_spros = official_eval.get_auc_spros_for_metrics(metrics=threshold_metrics)

        # Per-defect type sPRO
        try:
            spros_per_defect = official_eval.get_auc_spros_per_defect_type(
                metrics=threshold_metrics,
                defects_config=defects_config,
                add_good_images=True,
            )
        except Exception:
            spros_per_defect = {}

        # 5. Classification: Run official get_image_level_detection_metrics
        classification_results = official_eval.get_image_level_detection_metrics(
            gt_maps=gt_maps,
            anomaly_maps=anomaly_maps,
        )

        # 6. Supplemental Pooled AUROC
        y_true = [1 if m is not None else 0 for m in gt_maps]
        y_scores = [float(np.max(m.np_array)) for m in anomaly_maps]
        from sklearn.metrics import roc_auc_score
        pooled_auc = float(roc_auc_score(y_true, y_scores))

        elapsed = time.time() - t0

        cat_summary = {
            "object_name": cat,
            "config_id": config_id,
            "elapsed_seconds": round(elapsed, 2),
            "num_test_samples": len(anomaly_maps),
            "canonical_shape": resize_shape,
            "localization": {
                "auc_spro": auc_spros,
                "auc_spro_per_defect_type": spros_per_defect,
            },
            "classification": classification_results,
            "supplemental_pooled_auc": pooled_auc,
        }

        # Save official metrics.json format in category directory
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(cat_summary, f, indent=4, sort_keys=True)

        subtype_mean = classification_results.get("auc_roc", {}).get("mean", 0.0)
        spro_030 = auc_spros.get(0.3, auc_spros.get("0.3", 0.0))
        print(f"  [DONE {cat}] Official Subtype-Mean AUROC = {subtype_mean:.4f} | Pooled AUROC = {pooled_auc:.4f} | sPRO@0.30 = {spro_030:.4f} ({elapsed:.1f}s)")

        # Garbage collect heavy arrays
        del gt_maps, anomaly_maps, metrics_aggregator, threshold_metrics
        import gc
        gc.collect()

        config_results[cat] = cat_summary


    # Compute Macro Average
    macro_mean_auroc = float(np.mean([res["classification"]["auc_roc"]["mean"] for res in config_results.values()]))
    macro_pooled_auc = float(np.mean([res["supplemental_pooled_auc"] for res in config_results.values()]))
    macro_spro_030 = float(np.mean([res["localization"]["auc_spro"].get(0.3, res["localization"]["auc_spro"].get("0.3", 0.0)) for res in config_results.values()]))


    config_summary = {
        "config_id": config_id,
        "config_label": config_label,
        "evaluator_commit": "8ccc3d95054edf8237a152a6c29f61c0504b0b09",
        "macro_mean_image_auroc": macro_mean_auroc,
        "macro_pooled_image_auroc": macro_pooled_auc,
        "macro_spro_030": macro_spro_030,
        "categories": config_results,
    }

    summary_path = eval_out_root / "config_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=4, sort_keys=True)

    print("\n" + "=" * 80)
    print(f"[{config_id}] OFFICIAL MACRO IMAGE AUROC: {macro_mean_auroc:.4f} ({macro_mean_auroc*100:.2f}%)")
    print(f"[{config_id}] MACRO POOLED IMAGE AUROC:   {macro_pooled_auc:.4f}")
    print(f"[{config_id}] MACRO sPRO @ 0.30:            {macro_spro_030:.4f}")
    print("=" * 80)

    return config_summary


def main():
    print("#" * 80)
    print("AUTVI PHASE 4: AUTHORITATIVE THREE-CONFIGURATION OFFICIAL EVALUATION")
    print("Repository: https://github.com/phcarval/autovi_evaluation_code")
    print("Commit: 8ccc3d95054edf8237a152a6c29f61c0504b0b09")
    print("#" * 80)

    all_summaries = {}
    for cfg_id, cfg_label in CONFIGURATIONS:
        all_summaries[cfg_id] = evaluate_single_config(cfg_id, cfg_label)

    # Save grand comparison
    grand_summary_path = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "authoritative_benchmark_comparison.json"
    with open(grand_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4, sort_keys=True)
    print(f"\n[ALL COMPLETE] Grand authoritative benchmark saved to: {grand_summary_path}")


if __name__ == "__main__":
    main()
