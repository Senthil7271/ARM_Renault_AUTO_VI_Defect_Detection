"""Runner script for executing the official AutoVI evaluation benchmark.

Integrates with:
  external/autovi_evaluation_code/evaluate_experiment.py (commit 8ccc3d95054edf8237a152a6c29f61c0504b0b09)

Handles:
- Windows platform compatibility (os.nice shim)
- Proper dataset directory binding
- Dual evaluation: official AUC-sPRO (localization) + official AUC-ROC (classification)
- Saving consolidated results and metric comparison
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root and official eval repo to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
OFFICIAL_REPO_DIR = WORKSPACE_ROOT / "external" / "autovi_evaluation_code"

# Add official eval repo to sys.path and set working directory so local 'src' resolves
os.chdir(str(OFFICIAL_REPO_DIR))
sys.path.insert(0, str(OFFICIAL_REPO_DIR))

# Windows compatibility shim for os.nice
if not hasattr(os, "nice"):
    setattr(os, "nice", lambda inc: 0)

from src.aggregation import MetricsAggregator
from src.image import AnomalyMap, DefectsConfig, GroundTruthMap
from src.util import get_auc_for_max_fpr
import evaluate_experiment as official_eval

# Non-intrusive Windows POSIX-path compatibility patch for official evaluator
_orig_test_rel = official_eval.get_available_test_image_rel_paths
_orig_gt_rel = official_eval.get_available_gt_map_rel_paths

def _posix_test_rel_paths(test_images_dir: str):
    for p in _orig_test_rel(test_images_dir):
        yield p.replace("\\", "/")

def _posix_gt_rel_paths(gt_dir: str):
    for p in _orig_gt_rel(gt_dir):
        yield p.replace("\\", "/")

official_eval.get_available_test_image_rel_paths = _posix_test_rel_paths
official_eval.get_available_gt_map_rel_paths = _posix_gt_rel_paths


def evaluate_category_official(
    category: str,
    dataset_base_dir: Path,
    anomaly_maps_dir: Path,
    output_dir: Path,
    curve_max_distance: float = 0.002,
    num_parallel_workers: int = None,
) -> dict:
    print(f"\n=======================================================")
    print(f"Running Official AutoVI Evaluation: {category}")
    print(f"Dataset Base Dir:   {dataset_base_dir}")
    print(f"Anomaly Maps Dir:   {anomaly_maps_dir}")
    print(f"Output Dir:         {output_dir}")
    print(f"Max Curve Distance: {curve_max_distance}")
    print(f"=======================================================")

    t0_eval = time.perf_counter()

    # 1. Resolve canonical resize shape
    resize_shape = official_eval.set_resize_shape(category)
    print(f"Official Canonical Shape (W, H): {resize_shape}")

    # 2. Read defects config
    defects_config_path = dataset_base_dir / category / "defects_config.json"
    if not defects_config_path.exists():
        raise FileNotFoundError(f"Missing defects_config.json at {defects_config_path}")

    with open(defects_config_path, "r", encoding="utf-8") as f:
        defects_list = json.load(f)
    defects_config = DefectsConfig.create_from_list(defects_list)
    print(f"Loaded {len(defects_config.entries)} defect configuration entries:")
    for d in defects_config.entries:
        print(f"  - {d.defect_name}: saturation={d.saturation_threshold} (relative={d.relative_saturation})")

    # 3. Read Ground Truth Maps & Anomaly Maps
    gt_dir = dataset_base_dir / category / "ground_truth"
    print(f"Reading ground truth from: {gt_dir}")
    gt_maps, anomaly_maps = official_eval.read_maps(
        gt_dir=str(gt_dir),
        anomaly_maps_test_dir=str(anomaly_maps_dir),
        defects_config=defects_config,
        resize_shape=resize_shape,
    )
    print(f"Loaded {len(anomaly_maps)} anomaly maps and {len(gt_maps)} ground-truth associations.")

    # 4. Run Official Localization Evaluation (sPRO)
    print("Computing sPRO curve via dynamic binary refinement...")
    metrics_aggregator = MetricsAggregator(
        gt_maps=gt_maps,
        anomaly_maps=anomaly_maps,
        parallel_workers=num_parallel_workers,
        parallel_niceness=19,
    )
    threshold_metrics = metrics_aggregator.run(curve_max_distance=curve_max_distance)

    # AUC sPRO values per integration limit
    auc_spros = official_eval.get_auc_spros_for_metrics(metrics=threshold_metrics)
    print("\n--- Official Localization Results (AUC-sPRO) ---")
    for limit, score in auc_spros.items():
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  AUC-sPRO @ max FPR {limit:<4}: {score_str}")

    # Per-defect-type AUC-sPRO
    try:
        spros_per_defect = official_eval.get_auc_spros_per_defect_type(
            metrics=threshold_metrics,
            defects_config=defects_config,
            add_good_images=True,
        )
    except Exception as e:
        print(f"[NOTICE] Per-defect sPRO calculation note: {e}")
        spros_per_defect = {}

    # 5. Run Official Classification Evaluation (AUC-ROC)
    print("\nComputing official image-level anomaly classification metrics...")
    try:
        classification_results = official_eval.get_image_level_detection_metrics(
            gt_maps=gt_maps, anomaly_maps=anomaly_maps
        )
        print("--- Official Classification Results (AUC-ROC) ---")
        for k, v in classification_results.get("auc_roc", {}).items():
            print(f"  Image AUC-ROC [{k}]: {v:.4f}")
    except Exception as e:
        print(f"[WARN] Error running official image classification evaluation: {e}")
        classification_results = {"auc_roc": {}}

    total_time = time.perf_counter() - t0_eval

    # 6. Assemble Full Evaluation Report
    consolidated_results = {
        "category": category,
        "evaluator_repo": "https://github.com/phcarval/autovi_evaluation_code",
        "evaluator_commit": "8ccc3d95054edf8237a152a6c29f61c0504b0b09",
        "total_evaluation_seconds": round(total_time, 2),
        "curve_max_distance": curve_max_distance,
        "localization": {
            "auc_spro": auc_spros,
            "auc_spro_per_defect_type": spros_per_defect,
        },
        "classification": classification_results,
    }

    # Save to output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "official_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(consolidated_results, f, indent=4, sort_keys=True)

    print(f"\n[DONE] Official evaluation completed in {total_time:.1f}s.")
    print(f"Results saved to: {out_json}")
    return consolidated_results


def main():
    parser = argparse.ArgumentParser(description="Execute official AutoVI benchmark evaluation")
    parser.add_argument("--category", type=str, default="pipe_clip", help="Category name or 'all'")
    parser.add_argument("--curve-max-distance", type=float, default=0.002, help="Curve refinement tolerance")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "official_autovi" / "evaluation"),
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--anomaly-maps-dir",
        type=str,
        default=None,
        help="Optional custom anomaly maps directory",
    )
    args = parser.parse_args()

    categories = [
        "pipe_clip",
        "pipe_staple",
        "engine_wiring",
        "tank_screw",
        "underbody_pipes",
        "underbody_screw",
    ] if args.category.lower() == "all" else [args.category]

    all_official_results = {}
    for cat in categories:
        dataset_base = WORKSPACE_ROOT / cat
        if args.anomaly_maps_dir:
            p_cand = Path(args.anomaly_maps_dir)
            anomaly_maps_p = p_cand if p_cand.is_absolute() else (WORKSPACE_ROOT / p_cand)
        else:
            anomaly_maps_p = PROJECT_ROOT / "outputs" / "official_autovi" / "anomaly_maps" / cat / "test"
        
        p_out = Path(args.output_dir)
        cat_out = (p_out if p_out.is_absolute() else (WORKSPACE_ROOT / p_out)) / cat

        if not anomaly_maps_p.exists():
            print(f"[ERROR] Anomaly maps not found at {anomaly_maps_p}. Run export_official_predictions.py first.")
            continue

        res = evaluate_category_official(
            category=cat,
            dataset_base_dir=dataset_base,
            anomaly_maps_dir=anomaly_maps_p,
            output_dir=cat_out,
            curve_max_distance=args.curve_max_distance,
            num_parallel_workers=None,
        )
        all_official_results[cat] = res

    # If all categories evaluated, save portfolio summary
    if len(all_official_results) > 1:
        summary_file = Path(args.output_dir) / "all_categories_official_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_official_results, f, indent=4, sort_keys=True)
        print(f"\n[ALL DONE] Portfolio summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
