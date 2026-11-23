"""Anomaly Detection Evaluation Metrics.

Computes:
- Image AUROC, Image Average Precision (PR-AUC)
- Pixel AUROC, Pixel Average Precision
- AUPRO (Area Under the Per-Region Overlap curve)
- Precision, Recall, F1, FPR, Confusion Matrix at calibrated threshold
- IoU and Dice at segmentation threshold
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_image_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Computes image-level ranking and thresholded classification metrics."""
    # Ensure binary labels
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Ranking metrics (independent of threshold)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = float("nan")

    try:
        precisions, recalls, _ = precision_recall_curve(y_true, y_score)
        ap = float(auc(recalls, precisions))
    except Exception:
        ap = float("nan")

    # Threshold-dependent metrics
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "image_auroc": auroc,
        "image_ap": ap,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "confusion_matrix": {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }
    }


def compute_pixel_metrics(
    masks_true: np.ndarray, maps_pred: np.ndarray, threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Computes pixel-level ranking and localization metrics.
    
    Args:
        masks_true: (N, H, W) binary ground-truth masks
        maps_pred: (N, H, W) continuous anomaly heatmaps
        threshold: Optional segmentation threshold
    """
    flat_true = masks_true.ravel().astype(int)
    flat_pred = maps_pred.ravel().astype(float)

    # Subsample if extremely large to prevent OOM
    max_pts = 10_000_000
    if len(flat_true) > max_pts:
        step = len(flat_true) // max_pts
        flat_true = flat_true[::step]
        flat_pred = flat_pred[::step]

    # Pixel AUROC and AP
    try:
        pixel_auroc = float(roc_auc_score(flat_true, flat_pred))
    except Exception:
        pixel_auroc = float("nan")

    try:
        precisions, recalls, _ = precision_recall_curve(flat_true, flat_pred)
        pixel_ap = float(auc(recalls, precisions))
    except Exception:
        pixel_ap = float("nan")

    res = {
        "pixel_auroc": pixel_auroc,
        "pixel_ap": pixel_ap,
    }

    if threshold is not None:
        bin_pred = (flat_pred >= threshold).astype(int)
        intersection = np.sum((flat_true == 1) & (bin_pred == 1))
        union = np.sum((flat_true == 1) | (bin_pred == 1))
        iou = float(intersection / union) if union > 0 else 1.0

        dice_denom = np.sum(flat_true) + np.sum(bin_pred)
        dice = float(2.0 * intersection / dice_denom) if dice_denom > 0 else 1.0

        res["pixel_iou"] = iou
        res["pixel_dice"] = dice

    return res


def compute_au_pro(
    masks_true: np.ndarray, maps_pred: np.ndarray, max_fpr: float = 0.3, num_thresholds: int = 50
) -> float:
    """Computes Area Under Per-Region Overlap (AUPRO) metric.
    
    AUPRO measures detection rate on individual connected anomaly components.
    """
    if masks_true.sum() == 0:
        return 0.0

    min_val, max_val = float(maps_pred.min()), float(maps_pred.max())
    thresholds = np.linspace(min_val, max_val, num_thresholds)

    # Label connected components in ground truth
    labeled_regions = []
    for i in range(len(masks_true)):
        labeled, num_features = label(masks_true[i])
        for region_id in range(1, num_features + 1):
            labeled_regions.append((i, labeled == region_id))

    if not labeled_regions:
        return 0.0

    pro_scores = []
    fpr_scores = []

    total_normal_pixels = np.sum(masks_true == 0)
    if total_normal_pixels == 0:
        return 0.0

    for th in reversed(thresholds):
        bin_pred = (maps_pred >= th).astype(int)

        # FPR on normal pixels
        fp_pixels = np.sum((bin_pred == 1) & (masks_true == 0))
        fpr = fp_pixels / total_normal_pixels

        # PRO across all connected defect regions
        region_overlaps = []
        for img_idx, reg_mask in labeled_regions:
            region_size = np.sum(reg_mask)
            overlap = np.sum((bin_pred[img_idx] == 1) & reg_mask)
            region_overlaps.append(overlap / region_size)

        pro = float(np.mean(region_overlaps))

        if fpr <= max_fpr:
            pro_scores.append(pro)
            fpr_scores.append(fpr)

    if not fpr_scores:
        return 0.0

    fpr_scores = np.array(fpr_scores)
    pro_scores = np.array(pro_scores)

    # Sort by FPR
    sort_idx = np.argsort(fpr_scores)
    fpr_scores = fpr_scores[sort_idx]
    pro_scores = pro_scores[sort_idx]

    # Prepend (0, 0) if first FPR > 0, or ensure curve begins at 0
    if fpr_scores[0] > 0.0:
        fpr_scores = np.insert(fpr_scores, 0, 0.0)
        pro_scores = np.insert(pro_scores, 0, 0.0)

    # Extend to max_fpr if max FPR in evaluation is below max_fpr
    if fpr_scores[-1] < max_fpr:
        fpr_scores = np.append(fpr_scores, max_fpr)
        pro_scores = np.append(pro_scores, pro_scores[-1])

    # Compute trapezoid area and normalize by max_fpr
    try:
        trap_area = np.trapezoid(pro_scores, fpr_scores)
    except AttributeError:
        # Fallback for older numpy
        trap_area = np.trapz(pro_scores, fpr_scores)
    au_pro = float(trap_area / max_fpr)
    return au_pro
