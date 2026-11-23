"""AutoVI Edge: Ground Truth Mask Alignment and Segmentation Evaluation Engine.

Provides:
1. Exact Ground-Truth defect mask loading from AutoVI dataset.
2. Pixel-level IoU and Dice metric calculation.
3. DSMBAD/Visionairy quad-view comparison maps (Input, GT Mask, Heatmap, Error Map).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


class GroundTruthEvaluator:
    """Evaluates spatial defect localization against authoritative ground-truth annotations."""

    def __init__(self, gt_root: Optional[Path] = None):
        if gt_root is None:
            self.gt_root = Path(__file__).resolve().parent.parent / "ground_truth"
        else:
            self.gt_root = Path(gt_root)

    def load_ground_truth_mask(
        self,
        category: str,
        filename: str,
        target_shape: Tuple[int, int] = (256, 256),
    ) -> Tuple[np.ndarray, bool]:
        """Loads combined ground-truth binary mask (H, W) uint8 {0, 255}.
        
        filename: e.g. "unclipped/0000.jpg" or "good/0001.jpg"
        """
        h_t, w_t = target_shape[:2]
        is_nominal = "good" in filename.lower() or "nominal" in filename.lower()

        if is_nominal:
            return np.zeros((h_t, w_t), dtype=np.uint8), False

        parts = filename.replace("\\", "/").split("/")
        if len(parts) >= 2:
            subtype = parts[0]
            stem = Path(parts[-1]).stem  # e.g. "0000"
        else:
            subtype = "defect"
            stem = Path(filename).stem

        # Search for mask in ground_truth/<category>/<subtype>/<stem>/*.png
        mask_dir = self.gt_root / category / subtype / stem
        if not mask_dir.exists():
            # Try 4-digit stem with zfill
            z_stem = stem.zfill(4)
            mask_dir = self.gt_root / category / subtype / z_stem

        if mask_dir.exists():
            mask_files = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
            if mask_files:
                combined = None
                for mf in mask_files:
                    m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        bin_m = (m > 0).astype(np.uint8) * 255
                        if combined is None:
                            combined = bin_m
                        else:
                            combined = np.bitwise_or(combined, bin_m)
                if combined is not None:
                    resized = cv2.resize(combined, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
                    return resized, True

        # Fallback if no specific mask directory: return zero mask
        return np.zeros((h_t, w_t), dtype=np.uint8), False

    def evaluate_localization(
        self,
        gt_mask: np.ndarray,
        pred_map_norm: np.ndarray,
        pixel_threshold: float = 0.5,
    ) -> Dict:
        """Computes pixel IoU, Dice coefficient, and segmentation error map."""
        h, w = gt_mask.shape[:2]
        if pred_map_norm.shape[:2] != (h, w):
            pred_map_norm = cv2.resize(pred_map_norm, (w, h), interpolation=cv2.INTER_LINEAR)

        pred_binary = (pred_map_norm >= pixel_threshold).astype(np.uint8) * 255
        gt_binary = (gt_mask > 0).astype(np.uint8) * 255

        gt_bool = gt_binary > 0
        pred_bool = pred_binary > 0

        intersection = np.logical_and(gt_bool, pred_bool).sum()
        union = np.logical_or(gt_bool, pred_bool).sum()

        iou = float(intersection / union) if union > 0 else (1.0 if not gt_bool.any() and not pred_bool.any() else 0.0)
        dice = float(2.0 * intersection / (gt_bool.sum() + pred_bool.sum())) if (gt_bool.sum() + pred_bool.sum()) > 0 else (1.0 if not gt_bool.any() and not pred_bool.any() else 0.0)

        # Create Error Map:
        # Green: True Positive (Correctly detected defect)
        # Blue: False Positive (Over-detection / false alarm)
        # Red: False Negative (Missed defect region)
        error_map = np.zeros((h, w, 3), dtype=np.uint8)
        # TP -> Green (0, 255, 0)
        error_map[np.logical_and(gt_bool, pred_bool)] = [0, 220, 0]
        # FP -> Blue / Cyan (255, 180, 0) in BGR
        error_map[np.logical_and(~gt_bool, pred_bool)] = [255, 120, 0]
        # FN -> Red (0, 0, 255)
        error_map[np.logical_and(gt_bool, ~pred_bool)] = [0, 0, 255]

        match_status = "EXCELLENT" if iou >= 0.5 else ("ACCEPTABLE" if iou >= 0.25 else ("MISALIGNED" if gt_bool.any() else "NOMINAL_MATCH"))

        return {
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "has_ground_truth": bool(gt_bool.any()),
            "match_status": match_status,
            "error_map": error_map,
            "pred_binary": pred_binary,
        }
