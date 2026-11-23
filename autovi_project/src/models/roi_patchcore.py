"""Region-of-Interest (ROI) & Geometry-Aware PatchCore for Structural Anomaly Detection.

Candidate B Implementation (AutoVI Phase 4):
- Automatic normal-only derivation of part Region-of-Interest (ROI) from nominal training samples.
- Multi-scale feature extraction (Layers 2 & 3) restricted to the active component envelope.
- Normalized 2D spatial coordinate conditioning: [f_ij; alpha * y_norm, alpha * x_norm].
- Elimination of peripheral background false alarms (which hijack 86% of peak scores in baseline).
- Strict normal-only threshold calibration (Q95 on held-out nominal validation images).
- Zero reliance on test defects or test ground-truth masks.
"""

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patchcore_model import PatchCoreModel


class ROIPatchCoreModel(PatchCoreModel):
    """Region-of-Interest & Spatial-Conditioned PatchCore Model."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        target_size: Tuple[int, int] = (256, 256),
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        dimension_reduction: int = 128,
        chunk_size: int = 8192,
        candidate_pool_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        weights_path: Optional[str] = None,
        roi_box: Optional[Tuple[float, float, float, float]] = None,  # (ymin, xmin, ymax, xmax) in [0, 1]
        coord_scale: float = 0.05,  # Weight for spatial coordinate features
        top_k_ratio: float = 0.05,  # Top fraction of ROI patches for image score
    ):
        super().__init__(
            backbone_name=backbone_name,
            layers=layers,
            target_size=target_size,
            coreset_ratio=coreset_ratio,
            num_neighbors=num_neighbors,
            dimension_reduction=dimension_reduction,
            chunk_size=chunk_size,
            candidate_pool_size=candidate_pool_size,
            device=device,
            weights_path=weights_path,
        )
        self.roi_box = roi_box  # (ymin, xmin, ymax, xmax)
        self.coord_scale = coord_scale
        self.top_k_ratio = top_k_ratio
        self.roi_grid_coords: Optional[Tuple[int, int, int, int]] = None

    @staticmethod
    def derive_normal_roi_from_paths(
        image_paths: List[Union[str, Path]],
        target_size: Tuple[int, int] = (256, 256),
        padding_ratio: float = 0.08,
    ) -> Tuple[float, float, float, float]:
        """Derives canonical part bounding envelope purely from normal training images.
        
        Uses adaptive Otsu thresholding and morphological filtering to segment
        foreground components from dark chassis/background, taking the union bounding box.
        """
        all_x1, all_y1, all_x2, all_y2 = [], [], [], []
        h_t, w_t = target_size

        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img_r = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            # High gradient / edge density regions contain the mechanical part
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(grad_x, grad_y)
            mag_norm = np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)

            _, thresh = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Central component focus window (20% to 80%)
            c_y1, c_y2 = int(0.20 * h_t), int(0.80 * h_t)
            c_x1, c_x2 = int(0.20 * w_t), int(0.80 * w_t)
            central_mask = closed[c_y1:c_y2, c_x1:c_x2]

            ys, xs = np.where(central_mask > 0)
            if len(xs) > 50:
                all_x1.append((xs.min() + c_x1) / float(w_t))
                all_x2.append((xs.max() + c_x1) / float(w_t))
                all_y1.append((ys.min() + c_y1) / float(h_t))
                all_y2.append((ys.max() + c_y1) / float(h_t))

        if len(all_x1) == 0:
            return (0.35, 0.35, 0.80, 0.80)

        # Robust percentiles across all training samples
        ymin = max(0.20, float(np.percentile(all_y1, 10)) - padding_ratio)
        xmin = max(0.20, float(np.percentile(all_x1, 10)) - padding_ratio)
        ymax = min(0.85, float(np.percentile(all_y2, 90)) + padding_ratio)
        xmax = min(0.85, float(np.percentile(all_x2, 90)) + padding_ratio)

        return (round(ymin, 3), round(xmin, 3), round(ymax, 3), round(xmax, 3))

    def set_roi(self, roi_box: Tuple[float, float, float, float]):
        """Sets the normalized ROI box: (ymin, xmin, ymax, xmax) in [0, 1]."""
        self.roi_box = roi_box
        self.roi_grid_coords = None

    def _resolve_grid_coords(self, h_grid: int, w_grid: int) -> Tuple[int, int, int, int]:
        """Maps normalized ROI [0, 1] to discrete feature grid coordinates."""
        if self.roi_box is None:
            return (0, 0, h_grid, w_grid)
        ymin, xmin, ymax, xmax = self.roi_box
        gy1 = max(0, int(np.floor(ymin * h_grid)))
        gx1 = max(0, int(np.floor(xmin * w_grid)))
        gy2 = min(h_grid, int(np.ceil(ymax * h_grid)))
        gx2 = min(w_grid, int(np.ceil(xmax * w_grid)))
        return (gy1, gx1, gy2, gx2)

    def extract_patch_features(
        self,
        images: torch.Tensor,
        roi_only: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extracts multi-scale patch features, optionally restricted to the ROI grid
        and augmented with spatial coordinate embeddings."""
        raw_patches, (h, w) = super().extract_patch_features(images)
        b = images.size(0)
        d = raw_patches.shape[1]
        patches_reshaped = raw_patches.view(b, h, w, d)

        # Append 2D spatial coordinate embeddings: (y_norm, x_norm)
        if self.coord_scale > 0.0:
            y_coords = torch.linspace(-1.0, 1.0, h, device=self.device).view(1, h, 1, 1).expand(b, h, w, 1)
            x_coords = torch.linspace(-1.0, 1.0, w, device=self.device).view(1, 1, w, 1).expand(b, h, w, 1)
            coord_feats = torch.cat([y_coords, x_coords], dim=-1) * self.coord_scale
            patches_reshaped = torch.cat([patches_reshaped, coord_feats], dim=-1)
            d += 2

        gy1, gx1, gy2, gx2 = self._resolve_grid_coords(h, w)

        if roi_only and self.roi_box is not None:
            roi_patches = patches_reshaped[:, gy1:gy2, gx1:gx2, :].reshape(-1, d)
            return roi_patches, (gy2 - gy1, gx2 - gx1)
        else:
            return patches_reshaped.view(-1, d), (h, w)

    def fit_memory_bank(
        self,
        train_loader: Any,
        verbose: bool = True,
        chunk_size: Optional[int] = None,
        candidate_pool_size: Optional[int] = None,
    ):
        """Fits memory bank restricted strictly to the normal ROI component patches."""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if candidate_pool_size is not None:
            self.candidate_pool_size = candidate_pool_size

        total_start = time.perf_counter()
        self.backbone.eval()

        # Step 1: Probe first batch to determine exact ROI patch dimensions
        first_batch = next(iter(train_loader))
        first_images = first_batch["image"].to(self.device)
        first_patches, grid_shape = self.extract_patch_features(first_images, roi_only=True)
        dim = first_patches.shape[1]
        patches_per_image = len(first_patches) // len(first_images)

        total_images = len(train_loader.dataset)
        total_patches = total_images * patches_per_image

        if verbose:
            print(f"[ROI Memory Bank] Normal images: {total_images} | ROI patches per image: {patches_per_image}")
            print(f"[ROI Memory Bank] Pre-allocating ROI memory: ({total_patches}, {dim}) float32 ({(total_patches * dim * 4) / (1024**2):.1f} MiB)")

        raw_memory = np.empty((total_patches, dim), dtype=np.float32)

        # Copy first batch
        n_first = len(first_patches)
        raw_memory[0:n_first] = first_patches.cpu().numpy()
        cursor = n_first
        del first_patches, first_images, first_batch

        # Fill remaining batches
        for idx, batch in enumerate(train_loader):
            if idx == 0:
                continue
            images = batch["image"].to(self.device)
            patches, _ = self.extract_patch_features(images, roi_only=True)
            n_p = len(patches)
            raw_memory[cursor : cursor + n_p] = patches.cpu().numpy()
            cursor += n_p
            del patches, images

        target_coreset_size = max(1, int(total_patches * self.coreset_ratio))
        if verbose:
            print(f"[ROI Memory Bank] Coreset ratio: {self.coreset_ratio:.3f} | Target coreset vectors: {target_coreset_size}")

        if target_coreset_size >= total_patches:
            self.memory_bank = raw_memory
        else:
            # Chunked random projection to lower dimension
            proj_dim = min(self.dimension_reduction, dim)
            rng = np.random.default_rng(42)
            proj_matrix = rng.standard_normal((dim, proj_dim)).astype(np.float32)
            proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)

            features_proj = np.empty((total_patches, proj_dim), dtype=np.float32)
            for p_start in range(0, total_patches, self.chunk_size):
                p_end = min(p_start + self.chunk_size, total_patches)
                features_proj[p_start:p_end] = np.dot(raw_memory[p_start:p_end], proj_matrix)

            del proj_matrix

            # Greedy k-Center Subsampling
            sub_indices = self._greedy_kcenter_subsampling(
                features_proj, target_coreset_size, verbose=verbose
            )

            self.memory_bank = np.ascontiguousarray(raw_memory[sub_indices])
            del raw_memory
            del features_proj
            gc.collect()

        # Step 3: Build NearestNeighbors searcher
        self.nn_searcher = NearestNeighbors(
            n_neighbors=self.num_neighbors, algorithm="auto", metric="euclidean", n_jobs=-1
        )
        self.nn_searcher.fit(self.memory_bank)
        if verbose:
            print(f"[ROI Memory Bank] Fitted NearestNeighbors in {time.perf_counter() - total_start:.2f}s.")

    def predict_anomaly(
        self,
        image_tensor: torch.Tensor,
        gaussian_sigma: float = 4.0,
    ) -> Tuple[float, np.ndarray]:
        """Inference for a single image tensor (1, 3, H, W).
        
        Returns:
            image_score: Top-k mean anomaly score computed within the ROI.
            anomaly_map: 2D spatial heatmap (H, W) smoothed with Gaussian filter.
        """
        self.backbone.eval()
        with torch.no_grad():
            full_patches, (h, w) = self.extract_patch_features(image_tensor, roi_only=False)
            full_patches_np = full_patches.cpu().numpy()

            # Query memory bank for all patches
            distances, _ = self.nn_searcher.kneighbors(full_patches_np, n_neighbors=1)
            dist_map = distances[:, 0].reshape(h, w)

            # Map ROI coordinates
            gy1, gx1, gy2, gx2 = self._resolve_grid_coords(h, w)
            roi_dist = dist_map[gy1:gy2, gx1:gx2]

            # Compute image-level score from Top-K ROI patches
            flat_roi = roi_dist.flatten()
            k = max(1, int(len(flat_roi) * self.top_k_ratio))
            top_k_scores = np.partition(flat_roi, -k)[-k:]
            image_score = float(np.mean(top_k_scores))

            # Upsample anomaly map to input image resolution
            target_h, target_w = self.target_size
            anomaly_map = cv2.resize(dist_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if gaussian_sigma > 0:
                anomaly_map = gaussian_filter(anomaly_map, sigma=gaussian_sigma)

            return image_score, anomaly_map

    def save_roi_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict] = None):
        """Saves ROI PatchCore model weights, memory bank, and calibration metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_type": "roi_patchcore",
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "target_size": self.target_size,
            "coreset_ratio": self.coreset_ratio,
            "num_neighbors": self.num_neighbors,
            "dimension_reduction": self.dimension_reduction,
            "roi_box": self.roi_box,
            "coord_scale": self.coord_scale,
            "top_k_ratio": self.top_k_ratio,
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "memory_bank": self.memory_bank,
            "metadata": metadata or {},
        }
        torch.save(state, path)

    @classmethod
    def load_roi_checkpoint(cls, path: Union[str, Path], device: str = "cpu") -> "ROIPatchCoreModel":
        """Loads and initializes an ROIPatchCoreModel from a saved checkpoint."""
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            backbone_name=state.get("backbone_name", "wide_resnet50_2"),
            layers=tuple(state.get("layers", ("layer2", "layer3"))),
            target_size=tuple(state.get("target_size", (256, 256))),
            coreset_ratio=state.get("coreset_ratio", 0.1),
            num_neighbors=state.get("num_neighbors", 9),
            dimension_reduction=state.get("dimension_reduction", 128),
            device=device,
            roi_box=state.get("roi_box", None),
            coord_scale=state.get("coord_scale", 0.05),
            top_k_ratio=state.get("top_k_ratio", 0.05),
        )
        model.threshold = state.get("threshold", None)
        model.calibrated = state.get("calibrated", False)
        model.memory_bank = state["memory_bank"]

        # Fit NearestNeighbors on restored memory bank
        model.nn_searcher = NearestNeighbors(
            n_neighbors=model.num_neighbors, algorithm="auto", metric="euclidean", n_jobs=-1
        )
        model.nn_searcher.fit(model.memory_bank)
        model.to(device)
        model.eval()
        return model
