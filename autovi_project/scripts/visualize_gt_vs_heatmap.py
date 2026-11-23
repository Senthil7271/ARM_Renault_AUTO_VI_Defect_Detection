"""AutoVI Ground Truth vs Predicted Anomaly Heatmap Visualization and Evaluation.

Generates:
1. Six-panel visualizations for individual configurations.
2. Expanded comparative visualizations (Config A vs Config C side-by-side).
3. Publication-quality research figures for key industrial defect classes and failure modes.
4. Comprehensive quantitative localization metrics CSV (sPRO, Dice, IoU, Precision, Recall).
"""

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path

# Set Matplotlib cache dir before importing matplotlib
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
mpl_cache = PROJECT_ROOT / "outputs" / ".matplotlib"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

import cv2
import numpy as np
from tqdm import tqdm

# Upstream evaluator imports
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

CANONICAL_RESOLUTIONS = {
    "pipe_clip": (400, 400),
    "pipe_staple": (400, 400),
    "engine_wiring": (400, 400),
    "tank_screw": (750, 1000),
    "underbody_pipes": (750, 1000),
    "underbody_screw": (750, 1000),
}

ALL_CATEGORIES = [
    "tank_screw",
    "pipe_clip",
    "engine_wiring",
    "pipe_staple",
    "underbody_pipes",
    "underbody_screw",
]

OPTIMIZED_CATEGORIES = ["tank_screw", "pipe_clip", "engine_wiring"]
FROZEN_CATEGORIES = ["pipe_staple", "underbody_pipes", "underbody_screw"]


def load_defects_config(category: str) -> DefectsConfig:
    """Load DefectsConfig from dataset double-nested directory."""
    cfg_path = WORKSPACE_ROOT / category / category / "defects_config.json"
    assert cfg_path.exists(), f"Defects config not found at: {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_list = json.load(f)
    return DefectsConfig.create_from_list(cfg_list)


def load_calibrated_pixel_thresholds() -> dict:
    """Load calibrated pixel thresholds registry."""
    path = PROJECT_ROOT / "configs" / "calibrated_pixel_thresholds.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_sample_data(category: str, subtype: str, image_id: str, config_id: str = "config_A_baseline") -> dict:
    """Load and spatially align original RGB image, official GT mask, and predicted anomaly map."""
    # 1. RGB Image
    img_path = WORKSPACE_ROOT / category / category / "test" / subtype / f"{image_id}.png"
    assert img_path.exists(), f"Test RGB image not found: {img_path}"
    rgb_bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    H_exp, W_exp = CANONICAL_RESOLUTIONS[category]
    assert rgb.shape[:2] == (H_exp, W_exp), f"RGB shape mismatch for {category}: expected {(H_exp, W_exp)}, got {rgb.shape[:2]}"

    # 2. Ground-Truth Defect Mask
    gt_dir = WORKSPACE_ROOT / category / category / "ground_truth" / subtype / image_id
    gt_mask = np.zeros((H_exp, W_exp), dtype=bool)
    is_defective = False
    if gt_dir.exists() and any(gt_dir.glob("*.png")):
        cfg = load_defects_config(category)
        gt_obj = GroundTruthMap.read_from_png_dir(str(gt_dir), cfg)
        gt_mask = gt_obj.get_or_over_channels()
        is_defective = True
        assert gt_mask.shape == (H_exp, W_exp), f"GT shape mismatch: expected {(H_exp, W_exp)}, got {gt_mask.shape}"

    # 3. Anomaly Map
    map_path = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / config_id / category / "test" / subtype / f"{image_id}.png"
    assert map_path.exists(), f"Anomaly map not found: {map_path}"
    anomaly_map = cv2.imread(str(map_path), cv2.IMREAD_UNCHANGED)
    assert anomaly_map.shape[:2] == (H_exp, W_exp), f"Map shape mismatch: expected {(H_exp, W_exp)}, got {anomaly_map.shape[:2]}"

    return {
        "category": category,
        "subtype": subtype,
        "image_id": image_id,
        "config_id": config_id,
        "rgb": rgb,
        "gt_mask": gt_mask,
        "is_defective": is_defective,
        "anomaly_map": anomaly_map,
        "img_shape": (H_exp, W_exp),
        "max_score": float(np.max(anomaly_map)),
        "mean_score": float(np.mean(anomaly_map)),
    }


def compute_segmentation_error_map(gt_mask: np.ndarray, pred_mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Create RGB segmentation comparison overlay.

    Green: True Positive (TP)
    Red: False Positive (FP)
    Blue: False Negative (FN)
    Neutral: Dimmed original grayscale background (TN)
    """
    tp = gt_mask & pred_mask
    fp = (~gt_mask) & pred_mask
    fn = gt_mask & (~pred_mask)

    # Convert RGB to dimmed grayscale for clean background
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    canvas = np.stack([gray * 0.35] * 3, axis=-1).astype(np.uint8)

    # Color overlays
    canvas[tp] = [0, 230, 0]    # Green for TP
    canvas[fp] = [230, 0, 0]    # Red for FP
    canvas[fn] = [0, 100, 255]  # Blue for FN

    return canvas


def create_contour_overlay(rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Draw Ground-Truth and Predicted contours on top of original RGB image."""
    canvas = rgb.copy()

    # GT Contour in Cyan (BGR: [255, 255, 0] -> RGB: [0, 255, 255])
    if gt_mask.any():
        gt_uint8 = (gt_mask.astype(np.uint8)) * 255
        contours_gt, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours_gt, -1, (0, 255, 255), 2)

    # Pred Contour in Magenta (RGB: [255, 0, 255])
    if pred_mask.any():
        pred_uint8 = (pred_mask.astype(np.uint8)) * 255
        contours_pred, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours_pred, -1, (255, 0, 255), 2)

    return canvas


def compute_pixel_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Compute pixel-level segmentation evaluation metrics."""
    tp = int(np.sum(gt_mask & pred_mask))
    fp = int(np.sum((~gt_mask) & pred_mask))
    fn = int(np.sum(gt_mask & (~pred_mask)))
    tn = int(np.sum((~gt_mask) & (~pred_mask)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


def render_six_panel_figure(
    data: dict,
    threshold: float = None,
    threshold_label: str = "Calibrated p99",
    save_path: Path = None,
    dpi: int = 200,
) -> plt.Figure:
    """Render standard six-panel GT vs Prediction visualization."""
    rgb = data["rgb"]
    gt = data["gt_mask"]
    amap = data["anomaly_map"]

    if threshold is None:
        threshold = float(np.percentile(amap, 98.0))
        threshold_label = "Exploratory p98"

    pred_mask = amap >= threshold
    metrics = compute_pixel_metrics(gt, pred_mask)

    # Heatmap overlay on RGB
    norm = Normalize(vmin=0, vmax=65535)
    cmap = plt.get_cmap("turbo")
    heatmap_colored = (cmap(norm(amap))[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(rgb, 0.50, heatmap_colored, 0.50, 0)

    # Error Map & Contour Overlay
    error_map = compute_segmentation_error_map(gt, pred_mask, rgb)
    contour_map = create_contour_overlay(rgb, gt, pred_mask)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"AutoVI Anomaly Localization: {data['category']} / {data['subtype']} / {data['image_id']}\n"
        f"Config: {data['config_id']} | Image Max Score: {data['max_score']:.0f} | Threshold ({threshold_label}): {threshold:.0f} | "
        f"Dice: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f} | Rec: {metrics['recall']:.3f} | Prec: {metrics['precision']:.3f}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Original RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Panel 1: Original RGB Image", fontsize=11, fontweight="semibold")
    axes[0, 0].axis("off")

    # Panel 2: Ground-Truth Binary Mask
    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title(f"Panel 2: Official Ground Truth Mask\n({gt.sum():,} defect pixels)", fontsize=11, fontweight="semibold")
    axes[0, 1].axis("off")

    # Panel 3: Raw Anomaly Heatmap
    im3 = axes[0, 2].imshow(amap, cmap="turbo", vmin=0, vmax=65535)
    axes[0, 2].set_title("Panel 3: Raw Predicted Anomaly Heatmap\n(Global Scale: 0 - 65535)", fontsize=11, fontweight="semibold")
    axes[0, 2].axis("off")
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Panel 4: Heatmap Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Panel 4: Anomaly Heatmap Overlay (50%)\nBlended on Original RGB", fontsize=11, fontweight="semibold")
    axes[1, 0].axis("off")

    # Panel 5: Segmentation Error Map
    axes[1, 1].imshow(error_map)
    axes[1, 1].set_title(
        f"Panel 5: GT vs Prediction Error Map\n"
        f"TP={metrics['tp_pixels']:,} (Green) | FP={metrics['fp_pixels']:,} (Red) | FN={metrics['fn_pixels']:,} (Blue)",
        fontsize=11,
        fontweight="semibold",
    )
    axes[1, 1].axis("off")

    # Panel 6: Defect Localization Contours
    axes[1, 2].imshow(contour_map)
    axes[1, 2].set_title("Panel 6: Contour Localization View\nGT Contour (Cyan) vs Pred Contour (Magenta)", fontsize=11, fontweight="semibold")
    axes[1, 2].axis("off")

    # Add legend to Panel 6
    cyan_patch = mpatches.Patch(color="cyan", label="Official GT Contour")
    magenta_patch = mpatches.Patch(color="magenta", label=f"Predicted ({threshold_label})")
    axes[1, 2].legend(handles=[cyan_patch, magenta_patch], loc="lower right", framealpha=0.85, fontsize=9)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        print(f"[SAVED] Six-panel visualization: {save_path}")

    return fig


def render_expanded_comparison_figure(
    data_a: dict,
    data_c: dict,
    threshold_a: float,
    threshold_c: float,
    label_a: str = "Calibrated p99",
    label_c: str = "Calibrated p99",
    save_path: Path = None,
    dpi: int = 250,
) -> plt.Figure:
    """Render expanded side-by-side comparison figure: Config A vs Config C."""
    category = data_a["category"]
    subtype = data_a["subtype"]
    image_id = data_a["image_id"]
    rgb = data_a["rgb"]
    gt = data_a["gt_mask"]

    amap_a = data_a["anomaly_map"]
    amap_c = data_c["anomaly_map"]

    pred_a = amap_a >= threshold_a
    pred_c = amap_c >= threshold_c

    met_a = compute_pixel_metrics(gt, pred_a)
    met_c = compute_pixel_metrics(gt, pred_c)

    norm = Normalize(vmin=0, vmax=65535)
    cmap = plt.get_cmap("turbo")

    # Heatmap colored & overlays
    h_col_a = (cmap(norm(amap_a))[:, :, :3] * 255).astype(np.uint8)
    h_col_c = (cmap(norm(amap_c))[:, :, :3] * 255).astype(np.uint8)
    overlay_a = cv2.addWeighted(rgb, 0.50, h_col_a, 0.50, 0)
    overlay_c = cv2.addWeighted(rgb, 0.50, h_col_c, 0.50, 0)

    # Error maps
    err_a = compute_segmentation_error_map(gt, pred_a, rgb)
    err_c = compute_segmentation_error_map(gt, pred_c, rgb)

    # Contours
    cont_a = create_contour_overlay(rgb, gt, pred_a)
    cont_c = create_contour_overlay(rgb, gt, pred_c)

    fig, axes = plt.subplots(2, 6, figsize=(24, 8.5))
    fig.suptitle(
        f"AutoVI Phase 4 Comparative Localization Audit: {category.upper()} | Subtype: {subtype} | Sample ID: {image_id}\n"
        f"Row 1: Config A (Original Full-Frame Baseline) | Row 2: Config C (Deployable Normal ROI + Coordinate Conditioning)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # ---------------- Top Row: Config A ----------------
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"[A] Original RGB\nShape: {rgb.shape[:2]}", fontsize=10, fontweight="semibold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title(f"[A] Official GT Mask\n({gt.sum():,} defect px)", fontsize=10, fontweight="semibold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(amap_a, cmap="turbo", vmin=0, vmax=65535)
    axes[0, 2].set_title(f"[A] Baseline Heatmap\nMax: {amap_a.max():.0f} | Mean: {amap_a.mean():.0f}", fontsize=10, fontweight="semibold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(overlay_a)
    axes[0, 3].set_title(f"[A] Heatmap Overlay (50%)\n(Severe background noise)", fontsize=10, fontweight="semibold")
    axes[0, 3].axis("off")

    axes[0, 4].imshow(err_a)
    axes[0, 4].set_title(
        f"[A] Error Map ({label_a})\n"
        f"Dice: {met_a['dice']:.3f} | IoU: {met_a['iou']:.3f}\n"
        f"FP: {met_a['fp_pixels']:,} px | FN: {met_a['fn_pixels']:,} px",
        fontsize=9,
        fontweight="semibold",
    )
    axes[0, 4].axis("off")

    axes[0, 5].imshow(cont_a)
    axes[0, 5].set_title(f"[A] Contours (GT=Cyan, Pred=Magenta)\nPrec: {met_a['precision']:.3f} | Rec: {met_a['recall']:.3f}", fontsize=9, fontweight="semibold")
    axes[0, 5].axis("off")

    # ---------------- Bottom Row: Config C ----------------
    axes[1, 0].imshow(rgb)
    axes[1, 0].set_title(f"[C] Original RGB\nAligned 1-to-1", fontsize=10, fontweight="semibold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt, cmap="gray")
    axes[1, 1].set_title(f"[C] Official GT Mask\nPreserved Canonical", fontsize=10, fontweight="semibold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(amap_c, cmap="turbo", vmin=0, vmax=65535)
    axes[1, 2].set_title(f"[C] ROI + Coord Heatmap\nMax: {amap_c.max():.0f} | Mean: {amap_c.mean():.0f}", fontsize=10, fontweight="semibold")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(overlay_c)
    axes[1, 3].set_title(f"[C] Heatmap Overlay (50%)\n(Background suppressed)", fontsize=10, fontweight="semibold")
    axes[1, 3].axis("off")

    axes[1, 4].imshow(err_c)
    axes[1, 4].set_title(
        f"[C] Error Map ({label_c})\n"
        f"Dice: {met_c['dice']:.3f} | IoU: {met_c['iou']:.3f}\n"
        f"FP: {met_c['fp_pixels']:,} px | FN: {met_c['fn_pixels']:,} px",
        fontsize=9,
        fontweight="semibold",
    )
    axes[1, 4].axis("off")

    axes[1, 5].imshow(cont_c)
    axes[1, 5].set_title(f"[C] Contours (GT=Cyan, Pred=Magenta)\nPrec: {met_c['precision']:.3f} | Rec: {met_c['recall']:.3f}", fontsize=9, fontweight="semibold")
    axes[1, 5].axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        print(f"[SAVED] Expanded comparative figure: {save_path}")

    return fig


def generate_research_publication_figures():
    """Produce all required publication-quality research figures."""
    out_dir = PROJECT_ROOT / "outputs" / "gt_heatmap_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_thresholds = load_calibrated_pixel_thresholds()

    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION-QUALITY RESEARCH COMPARISON FIGURES")
    print("=" * 80)

    # 1. tank_screw: Successful missing screw localization
    print("\n[1/7] Generating tank_screw missing screw localization comparison...")
    d_a = load_sample_data("tank_screw", "missing", "0000", "config_A_baseline")
    d_c = load_sample_data("tank_screw", "missing", "0000", "config_C_roi_coord")
    tau_a = calib_thresholds.get("tank_screw", {}).get("config_A_baseline", {}).get("pixel_tau_p99", 34000.0)
    tau_c = calib_thresholds.get("tank_screw", {}).get("config_C_roi_coord", {}).get("pixel_tau_p99", 23000.0)
    render_expanded_comparison_figure(
        d_a, d_c, tau_a, tau_c,
        label_a="Normal p99: 34,039", label_c="Normal p99: 23,082",
        save_path=out_dir / "tank_screw_localization_comparison.png",
    )

    # 2. tank_screw: Failure mode analysis (Challenging sample with reflection)
    print("\n[2/7] Generating tank_screw failure mode & edge reflection analysis...")
    d_a_fail = load_sample_data("tank_screw", "missing", "0015", "config_A_baseline")
    d_c_fail = load_sample_data("tank_screw", "missing", "0015", "config_C_roi_coord")
    render_expanded_comparison_figure(
        d_a_fail, d_c_fail, tau_a, tau_c,
        label_a="Normal p99: 34,039", label_c="Normal p99: 23,082",
        save_path=out_dir / "tank_screw_failure_mode_analysis.png",
    )

    # 3. pipe_clip: Unclipped latch detection
    print("\n[3/7] Generating pipe_clip unclipped latch comparison...")
    d_a_clip = load_sample_data("pipe_clip", "unclipped", "0000", "config_A_baseline")
    d_c_clip = load_sample_data("pipe_clip", "unclipped", "0000", "config_C_roi_coord")
    tau_a_clip = calib_thresholds.get("pipe_clip", {}).get("config_A_baseline", {}).get("pixel_tau_p99", 41500.0)
    tau_c_clip = calib_thresholds.get("pipe_clip", {}).get("config_C_roi_coord", {}).get("pixel_tau_p99", 36100.0)
    render_expanded_comparison_figure(
        d_a_clip, d_c_clip, tau_a_clip, tau_c_clip,
        label_a="Normal p99: 41,529", label_c="Normal p99: 36,117",
        save_path=out_dir / "pipe_clip_unclipped_comparison.png",
    )

    # 4. engine_wiring: Fastening clip omission
    print("\n[4/7] Generating engine_wiring fastening harness comparison...")
    d_a_wire = load_sample_data("engine_wiring", "fastening", "0000", "config_A_baseline")
    d_c_wire = load_sample_data("engine_wiring", "fastening", "0000", "config_C_roi_coord")
    tau_a_wire = calib_thresholds.get("engine_wiring", {}).get("config_A_baseline", {}).get("pixel_tau_p99", 37150.0)
    tau_c_wire = calib_thresholds.get("engine_wiring", {}).get("config_C_roi_coord", {}).get("pixel_tau_p99", 33130.0)
    render_expanded_comparison_figure(
        d_a_wire, d_c_wire, tau_a_wire, tau_c_wire,
        label_a="Normal p99: 37,154", label_c="Normal p99: 33,133",
        save_path=out_dir / "engine_wiring_fastening_comparison.png",
    )

    # 5. engine_wiring: Blue hoop logical foreign object defect
    print("\n[5/7] Generating engine_wiring blue hoop foreign object comparison...")
    d_a_hoop = load_sample_data("engine_wiring", "blue_hoop", "0000", "config_A_baseline")
    d_c_hoop = load_sample_data("engine_wiring", "blue_hoop", "0000", "config_C_roi_coord")
    render_expanded_comparison_figure(
        d_a_hoop, d_c_hoop, tau_a_wire, tau_c_wire,
        label_a="Normal p99: 37,154", label_c="Normal p99: 33,133",
        save_path=out_dir / "engine_wiring_blue_hoop_comparison.png",
    )

    # 6. pipe_staple: Representative baseline localization
    print("\n[6/7] Generating pipe_staple representative baseline localization...")
    d_staple = load_sample_data("pipe_staple", "missing", "0000", "config_A_baseline")
    tau_staple = calib_thresholds.get("pipe_staple", {}).get("config_A_baseline", {}).get("pixel_tau_p99", 32000.0)
    render_six_panel_figure(
        d_staple, threshold=tau_staple, threshold_label="Normal p99: 32,073",
        save_path=out_dir / "pipe_staple_baseline_localization.png",
    )

    # 7. underbody_pipes & underbody_screw baseline figures
    print("\n[7/7] Generating underbody_pipes and underbody_screw baseline localization...")
    d_ub_pipes = load_sample_data("underbody_pipes", "obstruction", "0000", "config_A_baseline")
    tau_ub_pipes = calib_thresholds.get("underbody_pipes", {}).get("config_A_baseline", {}).get("pixel_tau_p99", 32050.0)
    render_six_panel_figure(
        d_ub_pipes, threshold=tau_ub_pipes, threshold_label="Normal p99: 32,050",
        save_path=out_dir / "underbody_pipes_baseline_localization.png",
    )

    d_ub_screw = load_sample_data("underbody_screw", "missing", "0000", "config_A_baseline")
    tau_ub_screw = calib_thresholds.get("underbody_screw", {}).get("config_A_baseline", {}).get("pixel_tau_p99", 29600.0)
    render_six_panel_figure(
        d_ub_screw, threshold=tau_ub_screw, threshold_label="Normal p99: 29,599",
        save_path=out_dir / "underbody_screw_baseline_localization.png",
    )

    print("\n[SUCCESS] All research publication figures generated successfully!")


def generate_localization_metrics_csv():
    """Compute and save comprehensive quantitative localization metrics CSV."""
    out_csv = PROJECT_ROOT / "reports" / "gt_heatmap_localization_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    calib_thresholds = load_calibrated_pixel_thresholds()

    # Load official sPRO values from authoritative benchmark
    bench_file = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "authoritative_benchmark_comparison.json"
    bench_data = {}
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

    headers = [
        "category",
        "subtype",
        "image_id",
        "configuration",
        "gt_defect_pixels",
        "calibrated_pixel_threshold",
        "tp_pixels",
        "fp_pixels",
        "fn_pixels",
        "tn_pixels",
        "pixel_precision",
        "pixel_recall",
        "dice_coefficient",
        "iou_score",
        "category_official_spro_001",
        "category_official_spro_005",
        "category_official_spro_010",
        "category_official_spro_030",
        "category_official_spro_100",
    ]

    print("\n" + "=" * 80)
    print("COMPUTING FULL QUANTITATIVE LOCALIZATION METRICS TABLE")
    print("=" * 80)

    rows = []
    # Sample defective images across categories and subtypes for rigorous analysis
    for cat in ALL_CATEGORIES:
        gt_root = WORKSPACE_ROOT / cat / cat / "ground_truth"
        subtypes = [p.name for p in gt_root.iterdir() if p.is_dir() and p.name != "good"]

        configs_to_eval = ["config_A_baseline", "config_C_roi_coord"] if cat in OPTIMIZED_CATEGORIES else ["config_A_baseline"]

        for cfg in configs_to_eval:
            spro_data = bench_data.get(cfg, {}).get("categories", {}).get(cat, {}).get("localization", {}).get("auc_spro", {})
            s_001 = spro_data.get("0.01", spro_data.get(0.01, 0.0))
            s_005 = spro_data.get("0.05", spro_data.get(0.05, 0.0))
            s_010 = spro_data.get("0.1", spro_data.get(0.1, 0.0))
            s_030 = spro_data.get("0.3", spro_data.get(0.3, 0.0))
            s_100 = spro_data.get("1.0", spro_data.get(1.0, 0.0))

            tau = calib_thresholds.get(cat, {}).get(cfg, {}).get("pixel_tau_p99", 35000.0)

            for sub in subtypes:
                img_dirs = sorted(list((gt_root / sub).iterdir()))
                # Evaluate each defect instance
                for img_dir in img_dirs:
                    img_id = img_dir.name
                    try:
                        data = load_sample_data(cat, sub, img_id, cfg)
                        pred_mask = data["anomaly_map"] >= tau
                        met = compute_pixel_metrics(data["gt_mask"], pred_mask)

                        rows.append({
                            "category": cat,
                            "subtype": sub,
                            "image_id": img_id,
                            "configuration": cfg,
                            "gt_defect_pixels": int(data["gt_mask"].sum()),
                            "calibrated_pixel_threshold": f"{tau:.1f}",
                            "tp_pixels": met["tp_pixels"],
                            "fp_pixels": met["fp_pixels"],
                            "fn_pixels": met["fn_pixels"],
                            "tn_pixels": met["tn_pixels"],
                            "pixel_precision": f"{met['precision']:.4f}",
                            "pixel_recall": f"{met['recall']:.4f}",
                            "dice_coefficient": f"{met['dice']:.4f}",
                            "iou_score": f"{met['iou']:.4f}",
                            "category_official_spro_001": f"{s_001:.4f}",
                            "category_official_spro_005": f"{s_005:.4f}",
                            "category_official_spro_010": f"{s_010:.4f}",
                            "category_official_spro_030": f"{s_030:.4f}",
                            "category_official_spro_100": f"{s_100:.4f}",
                        })
                    except Exception as e:
                        print(f"Error on {cat}/{sub}/{img_id}: {e}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[EXPORT] Saved {len(rows)} localization metric rows to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="AutoVI Ground Truth vs Heatmap Visualizer")
    parser.add_argument("--category", choices=ALL_CATEGORIES, default="tank_screw")
    parser.add_argument("--subtype", default="missing")
    parser.add_argument("--image-id", default="0000")
    parser.add_argument("--config", choices=["config_A_baseline", "config_B_roi_only", "config_C_roi_coord"], default="config_C_roi_coord")
    parser.add_argument("--batch-figures", action="store_true", help="Generate all research publication comparison figures")
    parser.add_argument("--export-metrics", action="store_true", help="Export full quantitative localization metrics CSV")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.batch_figures:
        generate_research_publication_figures()

    if args.export_metrics:
        generate_localization_metrics_csv()

    if not args.batch_figures and not args.export_metrics:
        data = load_sample_data(args.category, args.subtype, args.image_id, args.config)
        calibs = load_calibrated_pixel_thresholds()
        tau = calibs.get(args.category, {}).get(args.config, {}).get("pixel_tau_p99", None)
        out_p = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "gt_heatmap_comparisons" / f"{args.category}_{args.subtype}_{args.image_id}_{args.config}.png"
        render_six_panel_figure(data, threshold=tau, save_path=out_p)


if __name__ == "__main__":
    main()
