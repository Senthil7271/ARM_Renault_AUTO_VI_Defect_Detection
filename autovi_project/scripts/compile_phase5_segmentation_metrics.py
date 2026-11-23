"""Compile full Phase 5 quantitative segmentation metrics table.

Generates:
reports/phase5_segmentation_metrics.csv
Includes best, median, and worst localization examples across active categories.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage1_gt_alignment_audit import load_sample_data, compute_pixel_metrics, load_calibrated_pixel_thresholds
from scripts.visualize_gt_vs_heatmap import OPTIMIZED_CATEGORIES, CANONICAL_RESOLUTIONS

def main():
    print("=" * 80)
    print("GENERATING REPORTS/PHASE5_SEGMENTATION_METRICS.CSV")
    print("=" * 80)

    calib_thresholds = load_calibrated_pixel_thresholds()
    bench_file = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "authoritative_benchmark_comparison.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    rows = []

    for cat in OPTIMIZED_CATEGORIES:
        gt_root = WORKSPACE_ROOT / cat / cat / "ground_truth"
        subtypes = [p.name for p in gt_root.iterdir() if p.is_dir() and p.name != "good"]
        spro_dict = bench_data["config_C_roi_coord"]["categories"][cat]["localization"]["auc_spro"]
        tau = calib_thresholds[cat]["config_C_roi_coord"]["pixel_tau_p99"]

        for sub in subtypes:
            img_dirs = sorted(list((gt_root / sub).iterdir()))
            for img_dir in img_dirs:
                img_id = img_dir.name
                data = load_sample_data(cat, sub, img_id, "config_C_roi_coord")
                pred_mask = data["anomaly_map"] >= tau
                met = compute_pixel_metrics(data["gt_mask"], pred_mask)

                gt_px = int(data["gt_mask"].sum())
                pred_px = int(pred_mask.sum())
                area_ratio = pred_px / gt_px if gt_px > 0 else 0.0
                fp_ratio = met["fp_pixels"] / pred_px if pred_px > 0 else 0.0

                rows.append({
                    "category": cat,
                    "subtype": sub,
                    "image_id": img_id,
                    "configuration": "config_C_roi_coord",
                    "calibrated_pixel_threshold": tau,
                    "gt_defect_pixels": gt_px,
                    "pred_defect_pixels": pred_px,
                    "area_ratio": round(area_ratio, 3),
                    "fp_pixel_ratio": round(fp_ratio, 4),
                    "tp_pixels": met["tp_pixels"],
                    "fp_pixels": met["fp_pixels"],
                    "fn_pixels": met["fn_pixels"],
                    "tn_pixels": met["tn_pixels"],
                    "pixel_precision": round(met["precision"], 4),
                    "pixel_recall": round(met["recall"], 4),
                    "dice_coefficient": round(met["dice"], 4),
                    "iou_score": round(met["iou"], 4),
                    "official_spro_001": round(spro_dict.get("0.01", spro_dict.get(0.01, 0.0)), 4),
                    "official_spro_005": round(spro_dict.get("0.05", spro_dict.get(0.05, 0.0)), 4),
                    "official_spro_010": round(spro_dict.get("0.1", spro_dict.get(0.1, 0.0)), 4),
                    "official_spro_030": round(spro_dict.get("0.3", spro_dict.get(0.3, 0.0)), 4),
                    "official_spro_100": round(spro_dict.get("1.0", spro_dict.get(1.0, 0.0)), 4),
                })

    df = pd.DataFrame(rows)

    # Add percentile rank and rank tier within each category
    df["rank_in_category"] = ""
    for cat in OPTIMIZED_CATEGORIES:
        mask = df["category"] == cat
        cat_df = df[mask].sort_values("dice_coefficient", ascending=False)
        n = len(cat_df)
        
        best_idx = cat_df.index[0]
        median_idx = cat_df.index[n // 2]
        worst_idx = cat_df.index[-1]

        df.loc[mask, "rank_in_category"] = "Typical"
        df.loc[best_idx, "rank_in_category"] = "Best (Top 1)"
        df.loc[median_idx, "rank_in_category"] = "Median"
        df.loc[worst_idx, "rank_in_category"] = "Worst"

    out_csv = PROJECT_ROOT / "reports" / "phase5_segmentation_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Exported {len(df)} rows to: {out_csv}")

    # Print Best, Median, Worst breakdown
    for cat in OPTIMIZED_CATEGORIES:
        print(f"\n=== {cat.upper()} PERFORMANCE BREAKDOWN ===")
        sub_df = df[df["category"] == cat]
        b = sub_df[sub_df["rank_in_category"].str.startswith("Best")].iloc[0]
        m = sub_df[sub_df["rank_in_category"] == "Median"].iloc[0]
        w = sub_df[sub_df["rank_in_category"] == "Worst"].iloc[0]

        print(f"  * BEST  : Subtype={b['subtype']:15s} ID={b['image_id']} | Dice={b['dice_coefficient']:.4f} | IoU={b['iou_score']:.4f} | Prec={b['pixel_precision']:.4f} | Rec={b['pixel_recall']:.4f} | Ratio={b['area_ratio']}x")
        print(f"  * MEDIAN: Subtype={m['subtype']:15s} ID={m['image_id']} | Dice={m['dice_coefficient']:.4f} | IoU={m['iou_score']:.4f} | Prec={m['pixel_precision']:.4f} | Rec={m['pixel_recall']:.4f} | Ratio={m['area_ratio']}x")
        print(f"  * WORST : Subtype={w['subtype']:15s} ID={w['image_id']} | Dice={w['dice_coefficient']:.4f} | IoU={w['iou_score']:.4f} | Prec={w['pixel_precision']:.4f} | Rec={w['pixel_recall']:.4f} | Ratio={w['area_ratio']}x")


if __name__ == "__main__":
    main()
