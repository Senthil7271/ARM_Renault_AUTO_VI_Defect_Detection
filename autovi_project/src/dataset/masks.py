"""Ground truth mask processor for AutoVI dataset.

Handles:
- AutoVI nested mask directory resolution: `ground_truth/<defect_type>/<image_id>/<mask_id>.png`
- Union/combining multiple defect masks for a single image
- Multiclass label preservation according to defects_config.json
- Nearest-neighbor interpolation for resizing without intermediate blurry artifacts
- Zero-mask generation for normal test/train samples
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


class MaskProcessor:
    """Processes and pairs AutoVI ground-truth masks with inspection images."""

    def __init__(self, target_size: Optional[Tuple[int, int]] = (256, 256)):
        """
        Args:
            target_size: (height, width) or (width, height) to resize masks to.
        """
        self.target_size = target_size

    @staticmethod
    def find_mask_paths(category_root: Union[str, Path], defect_type: str, image_stem: str) -> List[Path]:
        """Finds all mask files corresponding to a test image.
        
        AutoVI stores masks as:
        `ground_truth/<defect_type>/<image_stem>/<mask_id>.png`
        """
        gt_dir = Path(category_root) / "ground_truth" / defect_type / image_stem
        if not gt_dir.exists():
            return []
        mask_files = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))
        return mask_files

    def load_combined_mask(
        self,
        mask_paths: List[Union[str, Path]],
        original_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and combines multiple masks for a single test image.
        
        Args:
            mask_paths: List of file paths to mask images.
            original_shape: (height, width) if no masks exist (e.g. good samples).
            
        Returns:
            binary_mask: np.ndarray (H, W), dtype uint8, values {0, 1}
            multiclass_mask: np.ndarray (H, W), dtype uint8, preserved pixel values
        """
        if not mask_paths:
            h, w = original_shape if original_shape is not None else (400, 400)
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            multiclass_mask = np.zeros((h, w), dtype=np.uint8)
            if self.target_size is not None:
                tw, th = self.target_size
                binary_mask = cv2.resize(binary_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                multiclass_mask = cv2.resize(multiclass_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            return binary_mask, multiclass_mask

        combined_binary = None
        combined_multiclass = None

        for path in mask_paths:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Handle 3-channel or 4-channel masks
            if img.ndim == 3:
                # If RGB/RGBA, take max across channels or 1st channel where values are positive
                mask_2d = np.max(img[..., :3], axis=-1)
            else:
                mask_2d = img

            bin_m = (mask_2d > 0).astype(np.uint8)

            if combined_binary is None:
                combined_binary = bin_m
                combined_multiclass = mask_2d.astype(np.uint8)
            else:
                combined_binary = np.bitwise_or(combined_binary, bin_m)
                # For multiclass, update pixels where new mask is positive
                combined_multiclass = np.where(mask_2d > 0, mask_2d, combined_multiclass)

        if combined_binary is None:
            h, w = original_shape if original_shape is not None else (400, 400)
            combined_binary = np.zeros((h, w), dtype=np.uint8)
            combined_multiclass = np.zeros((h, w), dtype=np.uint8)

        # Nearest-neighbor resize to avoid floating/blurred intermediate labels
        if self.target_size is not None:
            tw, th = self.target_size
            combined_binary = cv2.resize(combined_binary, (tw, th), interpolation=cv2.INTER_NEAREST)
            combined_multiclass = cv2.resize(combined_multiclass, (tw, th), interpolation=cv2.INTER_NEAREST)

        return combined_binary, combined_multiclass
