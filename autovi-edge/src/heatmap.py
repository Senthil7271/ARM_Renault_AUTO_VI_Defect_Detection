"""AutoVI Edge: Real-Time Anomaly Heatmap and Visualization Engine.

Transforms raw patch anomaly distances into calibrated, smoothed 2D heatmaps:
1. Bilinear upsampling to target display resolution.
2. 2D Gaussian spatial smoothing.
3. Deployable Normal ROI masking and 2D spatial coordinate falloff weighting (active categories).
4. False-color rendering (JET / TURBO / VIRIDIS) blended with live camera frames.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """Generates production-grade visual anomaly overlays for edge inspection."""

    def __init__(self, roi_configs: Optional[dict] = None):
        self.roi_configs = roi_configs or {}
        self.colormap = cv2.COLORMAP_JET
        self.alpha = 0.65  # Base frame weight
        self.beta = 0.35   # Heatmap overlay weight

    def set_colormap(self, colormap_name: str):
        maps = {
            "jet": cv2.COLORMAP_JET,
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "inferno": cv2.COLORMAP_INFERNO,
            "plasma": cv2.COLORMAP_PLASMA,
        }
        self.colormap = maps.get(colormap_name.lower(), cv2.COLORMAP_JET)

    def process(
        self,
        raw_distances: np.ndarray,      # (32, 32) or (1024,)
        category: str,
        target_shape: Tuple[int, int],  # (H, W) of camera frame
        category_min_max: Tuple[float, float], # (global_min, global_max)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Processes raw patch distances into normalized float heatmap and uint8 visualization array."""
        h_out, w_out = target_shape
        if raw_distances.ndim == 1:
            raw_2d = raw_distances.reshape(32, 32)
        else:
            raw_2d = raw_distances

        # 1. Bilinear Upsampling
        upsampled = cv2.resize(raw_2d, (w_out, h_out), interpolation=cv2.INTER_LINEAR)

        # 2. Gaussian Smoothing (sigma=4.0)
        smoothed = gaussian_filter(upsampled, sigma=4.0)

        # 3. Global Dynamic Range Normalization
        all_min, all_max = category_min_max
        rng = (all_max - all_min) if (all_max > all_min) else 1.0
        norm_map = np.clip((smoothed - all_min) / rng, 0.0, 1.0)

        # 4. Deployable ROI Masking & Spatial Coordinate Conditioning (Active Categories)
        if category in self.roi_configs:
            box = self.roi_configs[category]["deployable_normal_only_roi"]["box"]
            # box format: [ymin, xmin, ymax, xmax] relative to [0, 1]
            y1 = int(box[0] * h_out)
            x1 = int(box[1] * w_out)
            y2 = int(box[2] * h_out)
            x2 = int(box[3] * w_out)

            # Spatial Gaussian weighting centered on ROI
            cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
            sigma_y = max(10, (y2 - y1) / 2.0)
            sigma_x = max(10, (x2 - x1) / 2.0)
            yy, xx = np.mgrid[0:h_out, 0:w_out]
            dist_sq = ((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2
            spatial_weight = np.exp(-0.2 * dist_sq).astype(np.float32)

            # Apply ROI mask
            roi_mask = np.zeros_like(norm_map)
            roi_mask[y1:y2, x1:x2] = norm_map[y1:y2, x1:x2]
            final_float_map = roi_mask * spatial_weight
        else:
            final_float_map = norm_map

        # 5. Convert to 8-bit heatmap
        uint8_map = np.clip(final_float_map * 255.0, 0, 255).astype(np.uint8)
        return final_float_map, uint8_map

    def render_overlay(
        self,
        frame_bgr: np.ndarray,
        uint8_map: np.ndarray,
        category: str,
        verdict: str,
        score: float,
        threshold: float,
    ) -> np.ndarray:
        """Renders blended false-color heatmap on frame with inspection HUD overlay."""
        h_f, w_f = frame_bgr.shape[:2]
        if uint8_map.shape[:2] != (h_f, w_f):
            uint8_map = cv2.resize(uint8_map, (w_f, h_f), interpolation=cv2.INTER_LINEAR)

        # Apply False Color Map
        color_heatmap = cv2.applyColorMap(uint8_map, self.colormap)

        # Alpha Blend
        blended = cv2.addWeighted(frame_bgr, self.alpha, color_heatmap, self.beta, 0)

        # Draw ROI Bounding Box if category is ROI-conditioned
        if category in self.roi_configs:
            box = self.roi_configs[category]["deployable_normal_only_roi"]["box"]
            y1, x1 = int(box[0] * h_f), int(box[1] * w_f)
            y2, x2 = int(box[2] * h_f), int(box[3] * w_f)
            cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(blended, "ROI", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Status HUD Banner
        colors = {
            "PASS": (0, 200, 0),             # Green
            "REVIEW_REQUIRED": (0, 180, 255), # Amber / Orange
            "DEFECT": (0, 0, 255),            # Red
            "REJECT": (0, 0, 220),            # Red (Industrial binary reject)
        }
        badge_color = colors.get(verdict, (200, 200, 200))

        # Top Banner
        cv2.rectangle(blended, (0, 0), (w_f, 44), (20, 20, 20), -1)
        cv2.rectangle(blended, (10, 8), (140, 36), badge_color, -1)
        cv2.putText(blended, verdict, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_text = f"Cat: {category} | Score: {score:.2f} | Tau: {threshold:.2f}"
        cv2.putText(blended, info_text, (155, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

        return blended
