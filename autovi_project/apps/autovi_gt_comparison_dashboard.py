"""AutoVI Ground Truth vs Predicted Anomaly Heatmap Visualization Dashboard.

An interactive Streamlit application for deep visual inspection and quantitative
segmentation evaluation of industrial anomaly detection models across AutoVI categories.
Updated for Phase 5 with pixel histograms, Phase 5 comparative figures, and
best/median/worst case diagnostics.
"""

import json
import os
import sys
import types
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import streamlit as st

# Setup Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Set Matplotlib cache directory
mpl_cache = PROJECT_ROOT / "outputs" / ".matplotlib"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)

# Upstream Evaluator Integration
OFFICIAL_EVAL_DIR = WORKSPACE_ROOT / "external" / "autovi_evaluation_code"
if not hasattr(os, "nice"):
    setattr(os, "nice", lambda inc: 0)

sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

# Ensure namespace package 'src' cleanly resolves to external/autovi_evaluation_code/src
src_eval_path = str(OFFICIAL_EVAL_DIR / "src")
if "src" not in sys.modules or not hasattr(sys.modules["src"], "__path__"):
    _src_mod = types.ModuleType("src")
    _src_mod.__path__ = [src_eval_path]
    sys.modules["src"] = _src_mod
else:
    if src_eval_path not in sys.modules["src"].__path__:
        sys.modules["src"].__path__.insert(0, src_eval_path)

from src.image import GroundTruthMap, DefectsConfig
from scripts.visualize_gt_vs_heatmap import (
    ALL_CATEGORIES,
    OPTIMIZED_CATEGORIES,
    FROZEN_CATEGORIES,
    CANONICAL_RESOLUTIONS,
    load_defects_config,
    load_calibrated_pixel_thresholds,
    load_sample_data,
    compute_segmentation_error_map,
    create_contour_overlay,
    compute_pixel_metrics,
)

# Page Configuration
st.set_page_config(
    page_title="AutoVI Anomaly Localization Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_available_subtypes(category: str):
    """Retrieve all available test subtypes for a category."""
    test_dir = WORKSPACE_ROOT / category / category / "test"
    if not test_dir.exists():
        return []
    subtypes = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
    return subtypes


@st.cache_data
def get_available_image_ids(category: str, subtype: str):
    """Retrieve all test image IDs for a category and subtype."""
    sub_dir = WORKSPACE_ROOT / category / category / "test" / subtype
    if not sub_dir.exists():
        return []
    img_files = sorted(list(sub_dir.glob("*.png")))
    return [p.stem for p in img_files]


@st.cache_data
def load_authoritative_benchmarks():
    """Load official authoritative benchmark metrics."""
    bench_file = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "authoritative_benchmark_comparison.json"
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_phase5_metrics():
    """Load Phase 5 quantitative segmentation metrics table."""
    p5_file = PROJECT_ROOT / "reports" / "phase5_segmentation_metrics.csv"
    if p5_file.exists():
        return pd.read_csv(p5_file)
    return pd.DataFrame()


def render_dashboard():
    st.title("🔍 AutoVI Anomaly Localization & Heatmap Dashboard")
    st.markdown(
        "**Quantitative ground truth comparison, segmentation error mapping, and spatial resolution verification.**"
    )

    # Prominent Disclaimer
    st.info(
        "⚠️ **Methodological Boundary Note**: Threshold sliders alter **display-only segmentation binarization** for "
        "visual inspection and pixel-level Dice/IoU calculation. Official model evaluation and rank metrics (AUC-sPRO, "
        "Subtype-mean AUROC) are evaluated strictly on the **immutable continuous anomaly maps**."
    )

    calib_thresholds = load_calibrated_pixel_thresholds()
    benchmarks = load_authoritative_benchmarks()
    p5_df = load_phase5_metrics()

    # Sidebar Controls
    st.sidebar.header("📁 Dataset & Model Selection")

    category = st.sidebar.selectbox("Category", ALL_CATEGORIES, index=0)
    is_optimized = category in OPTIMIZED_CATEGORIES

    # Case Navigation shortcut
    if is_optimized and not p5_df.empty:
        cat_p5 = p5_df[p5_df["category"] == category]
        case_options = ["Custom Selection", "Jump to Best (Top Dice)", "Jump to Median Case", "Jump to Worst Case"]
        case_pick = st.sidebar.selectbox("Quick Benchmark Jump", case_options, index=0)
    else:
        case_pick = "Custom Selection"

    available_subtypes = get_available_subtypes(category)
    def_subtypes = [s for s in available_subtypes if s != "good"]
    default_sub = def_subtypes[0] if def_subtypes else available_subtypes[0]

    if case_pick == "Jump to Best (Top Dice)":
        target_row = cat_p5[cat_p5["rank_in_category"].str.startswith("Best")].iloc[0]
        subtype = target_row["subtype"]
        image_id = str(target_row["image_id"]).zfill(4)
        st.sidebar.success(f"Loaded Best Case: {subtype}/{image_id} (Dice: {target_row['dice_coefficient']:.3f})")
    elif case_pick == "Jump to Median Case":
        target_row = cat_p5[cat_p5["rank_in_category"] == "Median"].iloc[0]
        subtype = target_row["subtype"]
        image_id = str(target_row["image_id"]).zfill(4)
        st.sidebar.info(f"Loaded Median Case: {subtype}/{image_id} (Dice: {target_row['dice_coefficient']:.3f})")
    elif case_pick == "Jump to Worst Case":
        target_row = cat_p5[cat_p5["rank_in_category"] == "Worst"].iloc[0]
        subtype = target_row["subtype"]
        image_id = str(target_row["image_id"]).zfill(4)
        st.sidebar.warning(f"Loaded Worst Case: {subtype}/{image_id} (Dice: {target_row['dice_coefficient']:.3f})")
    else:
        subtype = st.sidebar.selectbox("Subtype", available_subtypes, index=available_subtypes.index(default_sub))
        available_images = get_available_image_ids(category, subtype)
        image_id = st.sidebar.selectbox("Image ID", available_images, index=0 if available_images else 0)

    # Configuration Mode
    if is_optimized:
        view_mode = st.sidebar.radio(
            "Visualization View Mode",
            [
                "Side-by-Side Comparison (Config A vs Config C)",
                "Single View: Config C (Deployable Normal ROI + Coord)",
                "Single View: Config A (Original Full-Frame Baseline)",
            ],
            index=0,
        )
    else:
        view_mode = "Single View: Config A (Original Full-Frame Baseline)"
        st.sidebar.info(f"Category **{category}** is a FROZEN category evaluated on original Config A baseline.")

    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Display & Segmentation Settings")

    cmap_choice = st.sidebar.selectbox(
        "Heatmap Colormap", ["turbo", "jet", "viridis", "magma", "inferno"], index=0
    )
    alpha = st.sidebar.slider("Heatmap Overlay Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

    # Threshold Mode
    cfg_key = "config_C_roi_coord" if "Config C" in view_mode else "config_A_baseline"
    p99_default = calib_thresholds.get(category, {}).get(cfg_key, {}).get("pixel_tau_p99", 35000.0)
    p95_default = calib_thresholds.get(category, {}).get(cfg_key, {}).get("pixel_tau_p95", 25000.0)

    thresh_mode = st.sidebar.radio(
        "Pixel Segmentation Threshold Source",
        ["Calibrated Normal p99 (Recommended)", "Calibrated Normal p95", "Manual Slider (Display Only)"],
        index=0,
    )

    if thresh_mode == "Calibrated Normal p99 (Recommended)":
        active_thresh = float(p99_default)
        thresh_label = f"Normal p99 ({active_thresh:.0f})"
    elif thresh_mode == "Calibrated Normal p95":
        active_thresh = float(p95_default)
        thresh_label = f"Normal p95 ({active_thresh:.0f})"
    else:
        active_thresh = float(st.sidebar.slider("Custom Threshold", min_value=0, max_value=65535, value=int(p99_default), step=500))
        thresh_label = f"Manual ({active_thresh:.0f})"

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### 💡 Three-Tier Threshold Hierarchy\n"
        "- **Continuous [0, 65535]**: Anomaly map density for AUC-sPRO ranking.\n"
        "- **Image Threshold ($\\tau_{95}$)**: Max score cut-off for Image Classification.\n"
        "- **Pixel Threshold ($\\tau_{\\text{pixel}}$)**: Cut-off for binary segmentation & Dice/IoU."
    )

    if not image_id:
        st.warning("No images available for selected parameters.")
        return

    # Tabs
    tab_inspect, tab_hist, tab_gallery, tab_guide = st.tabs([
        "🔬 Interactive Inspection",
        "📊 Score Distribution & Histograms",
        "🖼️ Research Figures Gallery",
        "📖 Methodological Guide",
    ])

    with tab_inspect:
        if view_mode.startswith("Side-by-Side"):
            # Load both A and C
            data_a = load_sample_data(category, subtype, image_id, "config_A_baseline")
            data_c = load_sample_data(category, subtype, image_id, "config_C_roi_coord")

            score_a = data_a["max_score"]
            score_c = data_c["max_score"]

            tau_pix_a = calib_thresholds.get(category, {}).get("config_A_baseline", {}).get("pixel_tau_p99", 35000.0)
            tau_pix_c = active_thresh if "Config C" in thresh_label else calib_thresholds.get(category, {}).get("config_C_roi_coord", {}).get("pixel_tau_p99", 30000.0)

            mask_pred_a = data_a["anomaly_map"] >= tau_pix_a
            mask_pred_c = data_c["anomaly_map"] >= tau_pix_c

            met_a = compute_pixel_metrics(data_a["gt_mask"], mask_pred_a)
            met_c = compute_pixel_metrics(data_c["gt_mask"], mask_pred_c)

            gt_px = int(data_a["gt_mask"].sum())
            pred_px_a = int(mask_pred_a.sum())
            pred_px_c = int(mask_pred_c.sum())

            # Metrics Banner
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GT Defect Area", f"{gt_px:,} px", "YES" if data_a["is_defective"] else "NO (Nominal)")
            c2.metric("Config A Pred Area", f"{pred_px_a:,} px", f"Ratio: {pred_px_a / max(1, gt_px):.2f}x | Dice: {met_a['dice']:.3f}")
            c3.metric("Config C Pred Area", f"{pred_px_c:,} px", f"Ratio: {pred_px_c / max(1, gt_px):.2f}x | Dice: {met_c['dice']:.3f}", delta=f"{met_c['dice'] - met_a['dice']:+.3f}")
            
            spro_data = benchmarks.get("config_C_roi_coord", {}).get("categories", {}).get(category, {}).get("localization", {}).get("auc_spro", {})
            s_030 = spro_data.get("0.3", spro_data.get(0.3, 0.0))
            c4.metric(f"Official sPRO@0.30 ({category})", f"{s_030 * 100:.2f}%")

            st.markdown("### Side-by-Side Spatial Localization & Error Comparison")

            # Display side-by-side columns
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Config A (Baseline Full-Frame)")
                norm = Normalize(vmin=0, vmax=65535)
                cmap = plt.get_cmap(cmap_choice)
                h_col_a = (cmap(norm(data_a["anomaly_map"]))[:, :, :3] * 255).astype(np.uint8)
                ovl_a = cv2.addWeighted(data_a["rgb"], 1.0 - alpha, h_col_a, alpha, 0)
                err_a = compute_segmentation_error_map(data_a["gt_mask"], mask_pred_a, data_a["rgb"])
                cnt_a = create_contour_overlay(data_a["rgb"], data_a["gt_mask"], mask_pred_a)

                st.image(ovl_a, caption="Config A Anomaly Overlay", use_container_width=True)
                st.image(err_a, caption=f"Config A Error Map (Green=TP, Red=FP, Blue=FN) | Threshold: {tau_pix_a:.0f}", use_container_width=True)
                st.image(cnt_a, caption="Config A Contours (Cyan=GT, Magenta=Pred)", use_container_width=True)

            with col_b:
                st.subheader("Config C (Deployable Normal ROI + Coord)")
                h_col_c = (cmap(norm(data_c["anomaly_map"]))[:, :, :3] * 255).astype(np.uint8)
                ovl_c = cv2.addWeighted(data_c["rgb"], 1.0 - alpha, h_col_c, alpha, 0)
                err_c = compute_segmentation_error_map(data_c["gt_mask"], mask_pred_c, data_c["rgb"])
                cnt_c = create_contour_overlay(data_c["rgb"], data_c["gt_mask"], mask_pred_c)

                st.image(ovl_c, caption="Config C Anomaly Overlay", use_container_width=True)
                st.image(err_c, caption=f"Config C Error Map (Green=TP, Red=FP, Blue=FN) | Threshold: {tau_pix_c:.0f}", use_container_width=True)
                st.image(cnt_c, caption="Config C Contours (Cyan=GT, Magenta=Pred)", use_container_width=True)

            # Detailed metrics table
            st.markdown("#### Quantitative Pixel Segmentation Metrics")
            df_comp = pd.DataFrame([
                {
                    "Metric": "Pixel Precision",
                    "Config A": f"{met_a['precision']:.4f}",
                    "Config C": f"{met_c['precision']:.4f}",
                    "Delta": f"{met_c['precision'] - met_a['precision']:+.4f}",
                },
                {
                    "Metric": "Pixel Recall",
                    "Config A": f"{met_a['recall']:.4f}",
                    "Config C": f"{met_c['recall']:.4f}",
                    "Delta": f"{met_c['recall'] - met_a['recall']:+.4f}",
                },
                {
                    "Metric": "Dice Coefficient (F1)",
                    "Config A": f"{met_a['dice']:.4f}",
                    "Config C": f"{met_c['dice']:.4f}",
                    "Delta": f"{met_c['dice'] - met_a['dice']:+.4f}",
                },
                {
                    "Metric": "IoU (Jaccard)",
                    "Config A": f"{met_a['iou']:.4f}",
                    "Config C": f"{met_c['iou']:.4f}",
                    "Delta": f"{met_c['iou'] - met_a['iou']:+.4f}",
                },
                {
                    "Metric": "Area Ratio (Predicted / GT)",
                    "Config A": f"{pred_px_a / max(1, gt_px):.2f}x",
                    "Config C": f"{pred_px_c / max(1, gt_px):.2f}x",
                    "Delta": f"{(pred_px_c - pred_px_a) / max(1, gt_px):+.2f}x",
                },
                {
                    "Metric": "False Positive Pixels (FP)",
                    "Config A": f"{met_a['fp_pixels']:,}",
                    "Config C": f"{met_c['fp_pixels']:,}",
                    "Delta": f"{met_c['fp_pixels'] - met_a['fp_pixels']:+,}",
                },
                {
                    "Metric": "True Positive Pixels (TP)",
                    "Config A": f"{met_a['tp_pixels']:,}",
                    "Config C": f"{met_c['tp_pixels']:,}",
                    "Delta": f"{met_c['tp_pixels'] - met_a['tp_pixels']:+,}",
                },
            ])
            st.table(df_comp)

        else:
            # Single configuration view
            cfg_id = "config_C_roi_coord" if "Config C" in view_mode else "config_A_baseline"
            data = load_sample_data(category, subtype, image_id, cfg_id)

            pred_mask = data["anomaly_map"] >= active_thresh
            metrics = compute_pixel_metrics(data["gt_mask"], pred_mask)

            gt_px = int(data["gt_mask"].sum())
            pred_px = int(pred_mask.sum())

            # Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ground Truth Defect", "YES" if data["is_defective"] else "NO", f"{gt_px:,} defect pixels")
            c2.metric("Image Peak Anomaly", f"{data['max_score']:.0f}", f"Mean: {data['mean_score']:.1f}")
            c3.metric("Pixel Dice / IoU", f"{metrics['dice']:.3f}", f"IoU: {metrics['iou']:.3f}")
            c4.metric("Area Ratio (Pred/GT)", f"{pred_px / max(1, gt_px):.2f}x", f"Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f}")

            st.markdown(f"### Six-Panel Visual Inspection ({data['config_id']})")

            # 6 panels
            norm = Normalize(vmin=0, vmax=65535)
            cmap = plt.get_cmap(cmap_choice)
            heatmap_colored = (cmap(norm(data["anomaly_map"]))[:, :, :3] * 255).astype(np.uint8)
            overlay = cv2.addWeighted(data["rgb"], 1.0 - alpha, heatmap_colored, alpha, 0)
            error_map = compute_segmentation_error_map(data["gt_mask"], pred_mask, data["rgb"])
            contour_map = create_contour_overlay(data["rgb"], data["gt_mask"], pred_mask)

            r1_c1, r1_c2, r1_c3 = st.columns(3)
            with r1_c1:
                st.image(data["rgb"], caption="1. Original Test RGB Image", use_container_width=True)
            with r1_c2:
                gt_display = (data["gt_mask"].astype(np.uint8) * 255)
                st.image(gt_display, caption="2. Official GT Binary Mask", use_container_width=True)
            with r1_c3:
                st.image(heatmap_colored, caption=f"3. Anomaly Heatmap ({cmap_choice})", use_container_width=True)

            r2_c1, r2_c2, r2_c3 = st.columns(3)
            with r2_c1:
                st.image(overlay, caption="4. RGB + Heatmap Alpha Overlay", use_container_width=True)
            with r2_c2:
                st.image(error_map, caption=f"5. Error Map (TP=Green, FP=Red, FN=Blue) @ {thresh_label}", use_container_width=True)
            with r2_c3:
                st.image(contour_map, caption="6. Contours (Cyan=GT, Magenta=Pred)", use_container_width=True)

    with tab_hist:
        st.header("📊 Anomaly Score Distribution & Receptive Field Halo Analysis")
        data_c = load_sample_data(category, subtype, image_id, "config_C_roi_coord")
        amap_c = data_c["anomaly_map"]
        gt_c = data_c["gt_mask"]

        # Extract defect vs background pixels inside ROI
        nonzero_mask = amap_c > 0
        inside_roi_scores = amap_c[nonzero_mask]
        defect_scores = amap_c[gt_c] if gt_c.any() else np.array([])
        bg_scores = amap_c[nonzero_mask & (~gt_c)]

        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        if len(bg_scores) > 0:
            ax_hist.hist(bg_scores, bins=60, alpha=0.6, color="gray", label=f"ROI Background Pixels (N={len(bg_scores):,})", density=True)
        if len(defect_scores) > 0:
            ax_hist.hist(defect_scores, bins=60, alpha=0.7, color="crimson", label=f"GT Defect Pixels (N={len(defect_scores):,})", density=True)

        ax_hist.axvline(active_thresh, color="blue", linestyle="--", linewidth=2, label=f"Active Threshold ({thresh_label})")
        ax_hist.set_title(f"Score Distribution: {category} / {subtype} / {image_id}", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel("16-Bit Anomaly Score [0, 65535]", fontsize=10)
        ax_hist.set_ylabel("Probability Density", fontsize=10)
        ax_hist.legend(loc="upper right", fontsize=9)
        ax_hist.grid(alpha=0.3)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        st.markdown(
            """
            **Receptive Field Halo Mechanism**:
            - Background pixels directly adjacent to the defect receive elevated feature distances due to the WideResNet receptive field ($\approx 60-120$ pixels).
            - This creates a smooth gradient between the peak defect score and the nominal floor.
            - Higher normal-percentile thresholds (e.g. p99.5) compress the halo, raising Precision and Dice without sacrificing image-level detection.
            """
        )

    with tab_gallery:
        st.header("🖼️ High-Resolution Research Publication Figures")

        figs = [
            ("phase5_localization_comparisons/tank_screw_oversegmentation_case_study.png", "Phase 5: Tank Screw Over-Segmentation Case Study (Best vs Median vs Worst)"),
            ("phase5_localization_comparisons/pipe_clip_unclipped_halo_analysis.png", "Phase 5: Pipe Clip Unclipped Latch Halo Analysis"),
            ("phase5_localization_comparisons/engine_wiring_false_positive_breakdown.png", "Phase 5: Engine Wiring Foreign Object vs Diffuse Displacement"),
            ("phase5_localization_comparisons/postprocessing_threshold_ablation.png", "Phase 5: Effect of Normal-Only Threshold Calibration"),
            ("gt_heatmap_comparisons/tank_screw_localization_comparison.png", "Phase 4: Tank Screw Missing Screw Localization (Config A vs Config C)"),
            ("gt_heatmap_comparisons/pipe_clip_unclipped_comparison.png", "Phase 4: Pipe Clip Unclipped Latch Localization"),
            ("gt_heatmap_comparisons/engine_wiring_fastening_comparison.png", "Phase 4: Engine Wiring Fastening Harness Localization"),
            ("gt_heatmap_comparisons/underbody_pipes_baseline_localization.png", "Phase 4: Underbody Pipes Baseline Localization"),
        ]

        fig_choice = st.selectbox("Select Figure to View", [f[1] for f in figs], index=0)
        selected_file = [f[0] for f in figs if f[1] == fig_choice][0]
        fig_path = PROJECT_ROOT / "outputs" / selected_file

        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
        else:
            st.error(f"Figure file not found: {fig_path}")

    with tab_guide:
        st.header("📖 Scientific Localization & Segmentation Guide")
        st.markdown(
            """
            ### Three-Tier Threshold Hierarchy in AutoVI Anomaly Detection

            1. **Continuous Anomaly Heatmaps ($A(x, y) \\in [0, 65535]$)**:
               - Anomaly score field exported as 16-bit PNG.
               - Used directly by the official evaluator for continuous ranking metrics:
                 - **AUC-sPRO** (saturated Per-Region Overlap at $\\text{FPR} = 0.01, 0.05, 0.10, 0.30, 1.00$).
                 - Area under the ROC curve without binarization.

            2. **Image-Level Decision Threshold ($\\tau_{\\text{img}}$)**:
               - Calibrated on nominal training/validation images to achieve 95% nominal recall (5% False Alarm Rate).
               - Binarizes the entire image: $\\max_{x,y} A(x, y) \\ge \\tau_{\\text{img}} \\implies \\text{Defective (FAIL)}$.
               - Controls the industrial line reject rate.

            3. **Pixel-Level Segmentation Threshold ($\\tau_{\\text{pixel}}$)**:
               - Calibrated on nominal test background distributions (e.g., 99th percentile of nominal heatmaps).
               - Binarizes pixel masks for spatial quality evaluation (Dice, IoU, Precision, Recall).
               - Enables automated bounding box generation and robotic guidance.
            """
        )


if __name__ == "__main__":
    render_dashboard()
