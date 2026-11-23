"""Orchestration script to export predictions and run official benchmark evaluation for all AutoVI categories.

Executes sequential end-to-end evaluation:
1. Feature extraction and anomaly map export (canonical resolutions, uint16 PNG + float32 NPY)
2. Official sPRO and image AUROC evaluation via official benchmark evaluator
3. Cross-metric comparison table generation (previous SageMaker vs official benchmark)
4. Operational confusion matrix & threshold calibration analysis
5. Per-defect subtype granularity analysis
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

CATEGORIES = [
    "pipe_clip",
    "pipe_staple",
    "engine_wiring",
    "tank_screw",
    "underbody_pipes",
    "underbody_screw",
]

HISTORICAL_SAGEMAKER_METRICS = {
    "pipe_clip": {
        "image_auroc": 0.7515,
        "pixel_auroc": 0.9724,
        "aupro": 0.9152,
        "calibrated_threshold": 0.9189,
        "optimal_f1": 0.6552,
        "recall_at_calibrated": 0.2324,
        "precision_at_calibrated": 0.9706,
        "specificity_at_calibrated": 0.9926,
    },
    "pipe_staple": {
        "image_auroc": 0.9270,
        "pixel_auroc": 0.9561,
        "aupro": 0.8632,
        "calibrated_threshold": 0.9859,
        "optimal_f1": 0.8268,
        "recall_at_calibrated": 1.0000,
        "precision_at_calibrated": 0.7048,
        "specificity_at_calibrated": 0.5794,
    },
    "engine_wiring": {
        "image_auroc": 0.8044,
        "pixel_auroc": 0.8697,
        "aupro": 0.7242,
        "calibrated_threshold": 0.8173,
        "optimal_f1": 0.7813,
        "recall_at_calibrated": 0.3447,
        "precision_at_calibrated": 0.9231,
        "specificity_at_calibrated": 0.9714,
    },
    "tank_screw": {
        "image_auroc": 0.7927,
        "pixel_auroc": 0.9900,
        "aupro": 0.9334,
        "calibrated_threshold": 0.8524,
        "optimal_f1": 0.7438,
        "recall_at_calibrated": 0.4421,
        "precision_at_calibrated": 0.9767,
        "specificity_at_calibrated": 0.9902,
    },
    "underbody_pipes": {
        "image_auroc": 0.9933,
        "pixel_auroc": 0.9931,
        "aupro": 0.9695,
        "calibrated_threshold": 0.8354,
        "optimal_f1": 0.9829,
        "recall_at_calibrated": 0.9728,
        "precision_at_calibrated": 0.9931,
        "specificity_at_calibrated": 0.9949,
    },
    "underbody_screw": {
        "image_auroc": 0.9892,
        "pixel_auroc": 0.9996,
        "aupro": 0.9987,
        "calibrated_threshold": 0.9701,
        "optimal_f1": 0.9912,
        "recall_at_calibrated": 1.0000,
        "precision_at_calibrated": 0.9825,
        "specificity_at_calibrated": 0.9774,
    },
}


def run_pipeline():
    out_base = PROJECT_ROOT / "outputs" / "official_autovi"
    eval_base = out_base / "evaluation"
    eval_base.mkdir(parents=True, exist_ok=True)

    all_official_metrics = {}

    for cat in CATEGORIES:
        print(f"\n{'='*70}")
        print(f"PROCESSING CATEGORY: {cat}")
        print(f"{'='*70}")

        cat_eval_file = eval_base / cat / "official_metrics.json"
        cat_anomaly_maps = out_base / "anomaly_maps" / cat / "test"

        # Step 1: Export predictions if not already exported
        if not cat_anomaly_maps.exists() or len(list(cat_anomaly_maps.rglob("*.png"))) == 0:
            print(f"[EXPORT] Exporting predictions for {cat} via isolated worker...")
            export_script = PROJECT_ROOT / "scripts" / "export_official_predictions.py"
            cmd = [
                sys.executable,
                str(export_script),
                "--category", cat,
                "--batch-size", "16",
            ]
            t0 = time.time()
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            print(f"[EXPORT] Finished export for {cat} in {time.time() - t0:.1f}s.")
        else:
            n_maps = len(list(cat_anomaly_maps.rglob("*.png")))
            print(f"[EXPORT] Anomaly maps already present ({n_maps} maps). Skipping inference.")

        # Step 2: Run official evaluation if not already evaluated
        if not cat_eval_file.exists():
            print(f"[EVAL] Running official evaluation for {cat} via isolated worker...")
            eval_script = PROJECT_ROOT / "scripts" / "run_official_evaluation.py"
            cmd = [
                sys.executable,
                str(eval_script),
                "--category", cat,
                "--curve-max-distance", "0.002",
            ]
            t0 = time.time()
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            print(f"[EVAL] Finished official evaluation for {cat} in {time.time() - t0:.1f}s.")

        # Load metrics
        with open(cat_eval_file, "r", encoding="utf-8") as f:
            all_official_metrics[cat] = json.load(f)

    # Save consolidated summary
    summary_path = eval_base / "all_categories_official_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_official_metrics, f, indent=4, sort_keys=True)
    print(f"\n[DONE] Saved consolidated summary to {summary_path}")

    # Step 3: Generate Stage 8 Comparison Table
    generate_comparison_tables(all_official_metrics)


def generate_comparison_tables(all_official_metrics: Dict):
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cat in CATEGORIES:
        hist = HISTORICAL_SAGEMAKER_METRICS.get(cat, {})
        off = all_official_metrics.get(cat, {})
        off_loc = off.get("localization", {}).get("auc_spro", {})
        off_cls = off.get("classification", {}).get("auc_roc", {})

        row = {
            "Category": cat,
            "Previous_Image_AUROC": hist.get("image_auroc", np.nan),
            "Official_Image_AUROC_Mean": off_cls.get("mean", np.nan),
            "Image_AUROC_Delta": off_cls.get("mean", np.nan) - hist.get("image_auroc", np.nan),
            "Previous_Pixel_AUROC": hist.get("pixel_auroc", np.nan),
            "Official_Pixel_AUROC": "N/A (Official omitted)",
            "Previous_AUPRO_Standard": hist.get("aupro", np.nan),
            "Official_AUC_sPRO_0.01": off_loc.get("0.01", np.nan),
            "Official_AUC_sPRO_0.05": off_loc.get("0.05", np.nan),
            "Official_AUC_sPRO_0.10": off_loc.get("0.1", np.nan),
            "Official_AUC_sPRO_0.30": off_loc.get("0.3", np.nan),
            "Official_AUC_sPRO_1.00": off_loc.get("1.0", np.nan),
            "Historical_Calibrated_Thresh": hist.get("calibrated_threshold", np.nan),
            "Historical_Recall_at_Thresh": hist.get("recall_at_calibrated", np.nan),
            "Historical_Precision_at_Thresh": hist.get("precision_at_calibrated", np.nan),
            "Historical_Specificity_at_Thresh": hist.get("specificity_at_calibrated", np.nan),
        }
        rows.append(row)

    # Compute Macro Average
    df = pd.DataFrame(rows)
    macro_row = {
        "Category": "Macro Average",
        "Previous_Image_AUROC": df["Previous_Image_AUROC"].mean(),
        "Official_Image_AUROC_Mean": df["Official_Image_AUROC_Mean"].mean(),
        "Image_AUROC_Delta": df["Image_AUROC_Delta"].mean(),
        "Previous_Pixel_AUROC": df["Previous_Pixel_AUROC"].mean(),
        "Official_Pixel_AUROC": "N/A",
        "Previous_AUPRO_Standard": df["Previous_AUPRO_Standard"].mean(),
        "Official_AUC_sPRO_0.01": df["Official_AUC_sPRO_0.01"].mean(),
        "Official_AUC_sPRO_0.05": df["Official_AUC_sPRO_0.05"].mean(),
        "Official_AUC_sPRO_0.10": df["Official_AUC_sPRO_0.10"].mean(),
        "Official_AUC_sPRO_0.30": df["Official_AUC_sPRO_0.30"].mean(),
        "Official_AUC_sPRO_1.00": df["Official_AUC_sPRO_1.00"].mean(),
        "Historical_Calibrated_Thresh": np.nan,
        "Historical_Recall_at_Thresh": df["Historical_Recall_at_Thresh"].mean(),
        "Historical_Precision_at_Thresh": df["Historical_Precision_at_Thresh"].mean(),
        "Historical_Specificity_at_Thresh": df["Historical_Specificity_at_Thresh"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([macro_row])], ignore_index=True)

    csv_path = reports_dir / "previous_vs_official_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[REPORTS] Saved {csv_path}")

    # Generate Markdown Table
    md_path = reports_dir / "previous_vs_official_metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AutoVI Benchmark: Historical vs Official Evaluation Protocol\n\n")
        f.write("Comparison of PatchCore models evaluated under historical SageMaker pipeline (standard Scikit-Learn / TorchMetrics)\n")
        f.write("versus the official AutoVI benchmark protocol (`external/autovi_evaluation_code`).\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n### Metric Definitions & Methodological Discrepancies\n\n")
        f.write("1. **Image AUROC**: Historical evaluation computed a single binary AUROC across all anomaly classes combined. The official evaluator computes separate one-vs-rest AUROCs per defect subtype against the normal (`good`) test images and averages them (`mean`).\n")
        f.write("2. **Pixel AUROC**: Historical evaluation reported standard pixel-level AUROC. The official benchmark intentionally omits Pixel AUROC, arguing that pixel AUROC is biased towards large defect areas and unrepresentative of component-level industrial inspection.\n")
        f.write("3. **AUPRO vs AUC-sPRO**: Historical AUPRO computed standard continuous trapezoidal integration up to FPR 0.3. The official benchmark computes saturated Per-Region Overlap (**sPRO**) with ground-truth component saturation thresholds $\\tau_k$ specified in `defects_config.json`, evaluated across multiple fixed operational FPR cutoffs (0.01, 0.05, 0.10, 0.30, 1.00).\n")
    print(f"[REPORTS] Saved {md_path}")


if __name__ == "__main__":
    run_pipeline()
