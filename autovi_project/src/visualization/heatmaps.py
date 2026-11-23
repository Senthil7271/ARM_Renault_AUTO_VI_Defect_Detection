"""Visualization module for defect localization and inspection reports.

Produces four-panel comparisons:
┌─────────────────┬─────────────────┐
│ Original image  │ Ground-truth    │
│                 │ defect mask     │
├─────────────────┼─────────────────┤
│ Predicted       │ Heatmap         │
│ binary mask     │ overlay         │
└─────────────────┴─────────────────┘

Categorizes results by:
- true_positive (correctly detected defect)
- false_negative (missed defect)
- false_positive (false alarm on good sample)
- true_negative (correctly verified normal sample)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Visualizer:
    """Creates four-panel engineering visualizations of anomaly detections."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_four_panel(
        self,
        orig_img_path: Union[str, Path],
        gt_mask: np.ndarray,
        pred_heatmap: np.ndarray,
        threshold: float,
        score: float,
        label_true: int,
        defect_type: str,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Renders and saves a 4-panel comparison figure."""
        orig_pil = Image.open(orig_img_path).convert("RGB")
        orig_np = np.array(orig_pil)
        h, w = orig_np.shape[:2]

        # Resize heatmap and mask to original image dimensions if needed
        if pred_heatmap.shape[:2] != (h, w):
            pred_heatmap = cv2.resize(pred_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        if gt_mask.shape[:2] != (h, w):
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # Predicted binary mask
        pred_bin = (pred_heatmap >= threshold).astype(np.uint8) * 255

        # Heatmap overlay
        norm_map = np.clip((pred_heatmap - (threshold * 0.5)) / (threshold * 1.5 + 1e-6), 0.0, 1.0)
        norm_uint8 = (norm_map * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
        color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_np, 0.65, color_map_rgb, 0.35, 0)

        # Determine outcome
        pred_label = 1 if score >= threshold else 0
        if label_true == 1 and pred_label == 1:
            outcome = "TRUE POSITIVE (Detected Defect)"
            folder = "true_positive"
        elif label_true == 1 and pred_label == 0:
            outcome = "FALSE NEGATIVE (Missed Defect)"
            folder = "false_negative"
        elif label_true == 0 and pred_label == 1:
            outcome = "FALSE POSITIVE (False Alarm)"
            folder = "false_positive"
        else:
            outcome = "TRUE NEGATIVE (Correct Normal)"
            folder = "true_negative"

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Panel 1: Original Image
        axes[0, 0].imshow(orig_np)
        axes[0, 0].set_title(f"Original ({defect_type})\nTrue: {'Anomaly' if label_true else 'Normal'}", fontsize=11)
        axes[0, 0].axis("off")

        # Panel 2: Ground-truth Mask
        axes[0, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("Ground-Truth Mask (Defect Region)", fontsize=11)
        axes[0, 1].axis("off")

        # Panel 3: Predicted Binary Mask
        axes[1, 0].imshow(pred_bin, cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title(f"Predicted Mask (Thresh: {threshold:.3f})\nScore: {score:.3f} | {outcome}", fontsize=11)
        axes[1, 0].axis("off")

        # Panel 4: Heatmap Overlay
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Heatmap Overlay (JET colormap)", fontsize=11)
        axes[1, 1].axis("off")

        plt.suptitle(f"Sample: {Path(orig_img_path).name} — {outcome}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            sub_dir = self.output_dir / folder
            sub_dir.mkdir(parents=True, exist_ok=True)
            save_path = sub_dir / f"{Path(orig_img_path).stem}_four_panel.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=130)
        plt.close(fig)
        return save_path
