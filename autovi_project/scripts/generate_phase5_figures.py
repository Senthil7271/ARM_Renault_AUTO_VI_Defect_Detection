"""Generate Phase 5 publication-quality localization & false-positive diagnostic figures.

Outputs saved to:
outputs/phase5_localization_comparisons/
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set Matplotlib cache dir before importing matplotlib
mpl_cache = PROJECT_ROOT / "outputs" / ".matplotlib"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np

from scripts.stage1_gt_alignment_audit import load_sample_data, compute_pixel_metrics, load_calibrated_pixel_thresholds
from scripts.visualize_gt_vs_heatmap import compute_segmentation_error_map, create_contour_overlay


def render_comparative_trio(samples: list, titles: list, out_path: Path, suptitle: str):
    """Render 3 side-by-side diagnostic columns: [RGB, Heatmap Overlay, Error Map, Contours]."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)

    norm = Normalize(vmin=0, vmax=65535)
    cmap = plt.get_cmap("turbo")

    row_labels = ["1. Original RGB", "2. Anomaly Overlay", "3. Error Map (TP=Grn, FP=Red, FN=Blu)", "4. Contours (Cyan=GT, Mag=Pred)"]

    for col_idx, (data, title) in enumerate(zip(samples, titles)):
        rgb = data["rgb"]
        gt = data["gt_mask"]
        amap = data["anomaly_map"]
        tau = data["tau"]

        pred_mask = amap >= tau
        met = compute_pixel_metrics(gt, pred_mask)

        # Overlay
        heatmap_colored = (cmap(norm(amap))[:, :, :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(rgb, 0.50, heatmap_colored, 0.50, 0)

        # Error map & contours
        error_map = compute_segmentation_error_map(gt, pred_mask, rgb)
        contour_map = create_contour_overlay(rgb, gt, pred_mask)

        axes[0, col_idx].imshow(rgb)
        axes[0, col_idx].set_title(f"{title}\nDice: {met['dice']:.3f} | Prec: {met['precision']:.3f} | Rec: {met['recall']:.3f}", fontsize=11, fontweight="semibold")
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(overlay)
        axes[1, col_idx].axis("off")

        axes[2, col_idx].imshow(error_map)
        axes[2, col_idx].axis("off")

        axes[3, col_idx].imshow(contour_map)
        axes[3, col_idx].axis("off")

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="lime", label="True Positive (TP)"),
        mpatches.Patch(color="red", label="False Positive (FP - Halo Over-segmentation)"),
        mpatches.Patch(color="#3388ff", label="False Negative (FN)"),
        mpatches.Patch(color="cyan", label="Cyan Contour: Ground Truth"),
        mpatches.Patch(color="magenta", label="Magenta Contour: Predicted"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    print("=" * 80)
    print("GENERATING PHASE 5 COMPARISON FIGURES")
    print("=" * 80)

    calib_thresholds = load_calibrated_pixel_thresholds()
    out_dir = PROJECT_ROOT / "outputs" / "phase5_localization_comparisons"

    # 1. Tank Screw Over-Segmentation Case Study (Best vs Median vs Worst)
    tau_ts = calib_thresholds["tank_screw"]["config_C_roi_coord"]["pixel_tau_p99"]
    ts_b = load_sample_data("tank_screw", "missing", "0088", "config_C_roi_coord")
    ts_b["tau"] = tau_ts
    ts_m = load_sample_data("tank_screw", "missing", "0026", "config_C_roi_coord")
    ts_m["tau"] = tau_ts
    ts_w = load_sample_data("tank_screw", "missing", "0086", "config_C_roi_coord")
    ts_w["tau"] = tau_ts

    render_comparative_trio(
        [ts_b, ts_m, ts_w],
        ["Best (ID 0088 | Ratio 5.8x)", "Median (ID 0026 | Ratio 7.4x)", "Worst (ID 0086 | Ratio 10.1x)"],
        out_dir / "tank_screw_oversegmentation_case_study.png",
        "AutoVI Phase 5: tank_screw Missing Screw Over-Segmentation & Receptive Field Halo Analysis",
    )

    # 2. Pipe Clip Unclipped Halo Analysis
    tau_pc = calib_thresholds["pipe_clip"]["config_C_roi_coord"]["pixel_tau_p99"]
    pc_b = load_sample_data("pipe_clip", "unclipped", "0102", "config_C_roi_coord")
    pc_b["tau"] = tau_pc
    pc_m = load_sample_data("pipe_clip", "unclipped", "0027", "config_C_roi_coord")
    pc_m["tau"] = tau_pc
    pc_w = load_sample_data("pipe_clip", "unclipped", "0104", "config_C_roi_coord")
    pc_w["tau"] = tau_pc

    render_comparative_trio(
        [pc_b, pc_m, pc_w],
        ["Best (ID 0102 | Dice 0.745)", "Median (ID 0027 | Dice 0.519)", "Worst (ID 0104 | Dice 0.165)"],
        out_dir / "pipe_clip_unclipped_halo_analysis.png",
        "AutoVI Phase 5: pipe_clip Unclipped Latch Localization & False Positive Edge Analysis",
    )

    # 3. Engine Wiring False Positive Breakdown
    tau_ew = calib_thresholds["engine_wiring"]["config_C_roi_coord"]["pixel_tau_p99"]
    ew_b = load_sample_data("engine_wiring", "cardboard", "0003", "config_C_roi_coord")
    ew_b["tau"] = tau_ew
    ew_m = load_sample_data("engine_wiring", "fastening", "0107", "config_C_roi_coord")
    ew_m["tau"] = tau_ew
    ew_w = load_sample_data("engine_wiring", "fastening", "0271", "config_C_roi_coord")
    ew_w["tau"] = tau_ew

    render_comparative_trio(
        [ew_b, ew_m, ew_w],
        ["Foreign Object (Cardboard 0003)", "Unfastened Harness (Median 0107)", "Diffuse Detachment (Worst 0271)"],
        out_dir / "engine_wiring_false_positive_breakdown.png",
        "AutoVI Phase 5: engine_wiring Discrete Foreign Object vs Diffuse Fastening Displacement",
    )

    # 4. Postprocessing Threshold Comparison (Baseline p99 vs Normal p99.5 vs Normal p99.8)
    sample_pc = load_sample_data("pipe_clip", "unclipped", "0027", "config_C_roi_coord")
    s1 = dict(sample_pc)
    s1["tau"] = 36117.0  # p99
    s2 = dict(sample_pc)
    s2["tau"] = 38656.0  # p99.5
    s3 = dict(sample_pc)
    s3["tau"] = 42934.0  # p99.8

    render_comparative_trio(
        [s1, s2, s3],
        ["Normal p99 (36,117 | Ratio 2.78x)", "Normal p99.5 (38,656 | Ratio 1.55x)", "Normal p99.8 (42,934 | Under-segmented)"],
        out_dir / "postprocessing_threshold_ablation.png",
        "AutoVI Phase 5: Effect of Normal-Only Threshold Calibration on Predicted Defect Contours",
    )

    print("\n[SUCCESS] All Phase 5 comparison figures generated!")


if __name__ == "__main__":
    main()
