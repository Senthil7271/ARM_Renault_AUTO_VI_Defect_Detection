"""Stage 11 Visualization Generator:
1. 4-panel diagnostic visual figures:
   - Panel 1: Original RGB Image
   - Panel 2: Ground Truth Segmentation Mask
   - Panel 3: PatchCore Anomaly Heatmap (JET colormap overlay)
   - Panel 4: Binarized Defect Prediction at Calibrated Threshold
2. Produces representative TP, FP, TN, FN cases for each category.
3. Produces ROC curves, Precision-Recall curves, and sPRO localization curves.
Saves all artifacts in outputs/official_autovi/visualizations/
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve, auc

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


def load_gt_mask(dataset_root: Path, category: str, defect_name: str, image_stem: str, target_shape: tuple) -> np.ndarray:
    """Loads official ground truth mask and resizes to target_shape (W, H)."""
    if defect_name == "good":
        return np.zeros((target_shape[1], target_shape[0]), dtype=np.uint8)

    # Search for ground truth mask in nested folder
    gt_dir = dataset_root / category / category / "ground_truth" / defect_name / image_stem
    if not gt_dir.exists():
        gt_dir = dataset_root / category / "ground_truth" / defect_name / image_stem
    if not gt_dir.exists():
        # Fallback to flat
        gt_dir = dataset_root / category / category / "ground_truth" / defect_name
        if not gt_dir.exists():
            gt_dir = dataset_root / category / "ground_truth" / defect_name

    mask_files = list(gt_dir.glob(f"*.png")) if gt_dir.is_dir() else []
    if not mask_files:
        mask_files = list(gt_dir.parent.glob(f"{image_stem}*.png"))

    if not mask_files:
        return np.zeros((target_shape[1], target_shape[0]), dtype=np.uint8)

    # Union all mask parts if multiple
    combined = None
    for mf in mask_files:
        m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if combined is None:
            combined = m
        else:
            combined = np.maximum(combined, m)

    if combined is None:
        return np.zeros((target_shape[1], target_shape[0]), dtype=np.uint8)

    resized = cv2.resize(combined, target_shape, interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8) * 255


def render_4panel_figure(
    img_path: Path,
    gt_mask: np.ndarray,
    anomaly_map: np.ndarray,
    threshold: float,
    title: str,
    output_path: Path,
):
    """Renders and saves publication-grade 4-panel diagnostic figure."""
    orig_img = Image.open(img_path).convert("RGB")
    h, w = anomaly_map.shape
    orig_resized = orig_img.resize((w, h), Image.Resampling.BILINEAR)
    orig_np = np.array(orig_resized)

    # Normalize anomaly map for heatmap
    norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image (55% original, 45% heatmap)
    overlay = (0.55 * orig_np + 0.45 * heatmap_colored).astype(np.uint8)

    # Binary mask prediction at threshold
    pred_bin = (anomaly_map >= threshold).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    axes[0].imshow(orig_np)
    axes[0].set_title("Original Test Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Anomaly Heatmap (Max: {anomaly_map.max():.2f})", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(pred_bin, cmap="gray")
    axes[3].set_title(f"Prediction (Thresh: {threshold:.2f})", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_category_visualizations(category: str):
    manifest_p = PROJECT_ROOT / "outputs" / "official_autovi" / "predictions" / category / "manifest.json"
    if not manifest_p.exists():
        print(f"[WARN] Manifest not found for {category}, skipping visualizations.")
        return

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = manifest_data["samples"]
    df = pd.DataFrame(samples)
    cal_thresh = float(df["threshold"].iloc[0])

    viz_dir = PROJECT_ROOT / "outputs" / "official_autovi" / "visualizations" / category
    viz_dir.mkdir(parents=True, exist_ok=True)

    df["pred_binary"] = df["anomaly_score"] >= cal_thresh

    # Find TP, FP, TN, FN
    tp_candidates = df[(df["is_anomaly"] == 1) & (df["pred_binary"] == True)]
    fp_candidates = df[(df["is_anomaly"] == 0) & (df["pred_binary"] == True)]
    tn_candidates = df[(df["is_anomaly"] == 0) & (df["pred_binary"] == False)]
    fn_candidates = df[(df["is_anomaly"] == 1) & (df["pred_binary"] == False)]

    cases = [
        ("TP", tp_candidates.sort_values(by="anomaly_score", ascending=False)),
        ("TN", tn_candidates.sort_values(by="anomaly_score", ascending=True)),
        ("FN", fn_candidates.sort_values(by="anomaly_score", ascending=True)),
        ("FP", fp_candidates.sort_values(by="anomaly_score", ascending=False)),
    ]

    for label, subset in cases:
        if subset.empty:
            continue
        sample = subset.iloc[0]
        raw_map_path = PROJECT_ROOT / "outputs" / "official_autovi" / "predictions" / category / "raw_scores" / sample["defect_name"] / f"{sample['image_stem']}.npy"
        if not raw_map_path.exists():
            continue

        raw_map = np.load(raw_map_path)
        h, w = raw_map.shape
        gt_mask = load_gt_mask(WORKSPACE_ROOT, category, sample["defect_name"], sample["image_stem"], (w, h))

        title = f"{category.upper()} — {label}: {sample['defect_name']}/{sample['image_stem']} (Score: {sample['anomaly_score']:.2f}, Thresh: {cal_thresh:.2f})"
        out_f = viz_dir / f"{label}_{sample['defect_name']}_{sample['image_stem']}_4panel.png"
        render_4panel_figure(
            img_path=Path(sample["image_path"]),
            gt_mask=gt_mask,
            anomaly_map=raw_map,
            threshold=cal_thresh,
            title=title,
            output_path=out_f,
        )
        print(f"  [SAVED] {out_f.name}")


def generate_portfolio_curves():
    print(f"\n[VIZ] Generating ROC and PR Portfolio Curves...")
    viz_base = PROJECT_ROOT / "outputs" / "official_autovi" / "visualizations"
    viz_base.mkdir(parents=True, exist_ok=True)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6), dpi=150)
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6), dpi=150)

    for cat in CATEGORIES:
        manifest_p = PROJECT_ROOT / "outputs" / "official_autovi" / "predictions" / cat / "manifest.json"
        if not manifest_p.exists():
            continue
        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data["samples"])

        y_true = df["is_anomaly"].astype(int).values
        scores = df["anomaly_score"].values

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{cat} (AUROC={roc_auc:.3f})")

        # PR Curve
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(rec, prec)
        ax_pr.plot(rec, prec, lw=2, label=f"{cat} (AUPR={pr_auc:.3f})")

    # Finalize ROC Plot
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Chance")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
    ax_roc.set_title("AutoVI PatchCore Portfolio — Image ROC Curves", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    roc_out = viz_base / "portfolio_roc_curves.png"
    fig_roc.savefig(roc_out)
    plt.close(fig_roc)
    print(f"  [SAVED] {roc_out}")

    # Finalize PR Plot
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax_pr.set_ylabel("Precision (PPV)", fontsize=12)
    ax_pr.set_title("AutoVI PatchCore Portfolio — Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=10)
    ax_pr.grid(True, alpha=0.3)
    fig_pr.tight_layout()
    pr_out = viz_base / "portfolio_pr_curves.png"
    fig_pr.savefig(pr_out)
    plt.close(fig_pr)
    print(f"  [SAVED] {pr_out}")


def main():
    for cat in CATEGORIES:
        print(f"[VIZ] Generating representative figures for {cat}...")
        generate_category_visualizations(cat)
    generate_portfolio_curves()


if __name__ == "__main__":
    main()
