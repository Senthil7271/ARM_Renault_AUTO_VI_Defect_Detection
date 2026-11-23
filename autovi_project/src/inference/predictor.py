"""Anomaly Predictor Module for single image and batch inference.

Generates:
- Original image
- Image anomaly score
- Calibrated normal/anomaly decision
- Continuous anomaly heatmap
- Heatmap overlay
- Binary anomaly mask
- Bounding boxes from localized regions
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from ..models.patchcore_model import PatchCoreModel
from .events import DefectEventLogger


class AnomalyPredictor:
    """Performs inference using a trained PatchCore model."""

    def __init__(
        self,
        model: Union[PatchCoreModel, str, Path],
        device: Optional[str] = None,
        event_logger: Optional[DefectEventLogger] = None,
        category: str = "unknown",
    ):
        if isinstance(model, (str, Path)):
            self.model = PatchCoreModel.load(model, device=device)
        else:
            self.model = model

        self.category = category
        self.event_logger = event_logger
        self.target_size = self.model.target_size

        self.transform = T.Compose([
            T.Resize(self.target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_visuals: bool = True,
    ) -> Dict[str, Any]:
        """Runs anomaly detection on a single image file."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        img_tensor = self.transform(pil_img).unsqueeze(0)  # (1, 3, H, W)

        scores, heatmaps = self.model.predict_batch(img_tensor)
        score = float(scores[0])
        heatmap = heatmaps[0]  # (H, W)

        # Decision
        threshold = self.model.threshold
        is_calibrated = self.model.calibrated
        is_anomaly = (score >= threshold) if is_calibrated else None

        # Resize heatmap back to original image size
        heatmap_orig = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Normalize heatmap globally using threshold or max
        norm_map = np.clip((heatmap_orig - (threshold * 0.5)) / (threshold * 1.5 + 1e-6), 0.0, 1.0)
        norm_map_uint8 = (norm_map * 255).astype(np.uint8)

        # Binary predicted mask at threshold
        bin_mask = (heatmap_orig >= threshold).astype(np.uint8) * 255

        # Bounding boxes from connected components in binary mask
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 25:  # Filter noise pixels
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": float(area)})

        # Color heatmap overlay
        color_heatmap = cv2.applyColorMap(norm_map_uint8, cv2.COLORMAP_JET)
        orig_np = np.array(pil_img)[:, :, ::-1]  # RGB to BGR for cv2
        overlay = cv2.addWeighted(orig_np, 0.6, color_heatmap, 0.4, 0)

        # Draw bounding boxes on overlay
        for bbox in bounding_boxes:
            cv2.rectangle(
                overlay,
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
                (0, 0, 255),
                2,
            )

        paths = {}
        if save_visuals and output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem

            hm_path = out_p / f"{stem}_heatmap.png"
            ov_path = out_p / f"{stem}_overlay.png"
            mk_path = out_p / f"{stem}_mask.png"

            cv2.imwrite(str(hm_path), color_heatmap)
            cv2.imwrite(str(ov_path), overlay)
            cv2.imwrite(str(mk_path), bin_mask)

            paths = {
                "heatmap_path": str(hm_path),
                "overlay_path": str(ov_path),
                "mask_path": str(mk_path),
            }

        event = None
        if is_anomaly and self.event_logger is not None:
            event = self.event_logger.generate_event(
                category=self.category,
                anomaly_score=score,
                threshold=threshold,
                is_anomaly=True,
                heatmap_path=paths.get("heatmap_path"),
                mask_path=paths.get("mask_path"),
                bounding_boxes=bounding_boxes,
            )

        return {
            "image_path": str(image_path),
            "anomaly_score": score,
            "threshold": threshold,
            "calibrated": is_calibrated,
            "is_anomaly": is_anomaly,
            "prediction": "anomaly" if is_anomaly else "normal" if is_calibrated else "uncalibrated",
            "bounding_boxes": bounding_boxes,
            "event": event,
            **paths,
        }
