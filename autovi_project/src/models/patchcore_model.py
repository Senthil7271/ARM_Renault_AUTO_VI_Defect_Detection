"""PatchCore Anomaly Detection Model implementation.

Based on Roth et al. (2021): "Towards Total Recall in Industrial Anomaly Detection"
Memory-Optimized Architecture:
- WideResNet-50-2 (default) feature extractor
- Intermediate activations from layer2 and layer3
- Spatial neighborhood aggregation via AvgPool2d(3, stride=1, padding=1)
- Multiscale feature concatenation and patch embedding
- Memory-bounded Greedy k-Center coreset subsampling:
    * Pre-allocated contiguous float32 feature matrix (no duplicate copies)
    * Precomputed row-wise squared Euclidean norms
    * Chunked matrix-vector dot products (zero temporary (N, 128) broadcasting allocations)
    * Non-recurrent center masking (min_dist_sq = -1.0)
    * Immediate garbage collection of raw features prior to NearestNeighbors fitting
- Exact nearest-neighbor memory-bank scoring
- Gaussian-smoothed 2D spatial anomaly map reconstruction
- Full state persistence and standalone inference reloading
"""

import gc
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter

try:
    import psutil
except ImportError:
    psutil = None


def _get_rss_mb() -> float:
    """Returns current process resident memory (RSS) in MiB."""
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    return 0.0


def _get_vram_mb() -> float:
    """Returns current process allocated CUDA VRAM in MiB."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    return 0.0


class PatchCoreModel(nn.Module):
    """Memory-bounded PatchCore model with streaming coreset selection and persistence."""

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
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = list(layers)
        self.target_size = target_size
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.dimension_reduction = dimension_reduction
        self.chunk_size = chunk_size
        self.candidate_pool_size = candidate_pool_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build backbone
        self.backbone = self._load_backbone(backbone_name, weights_path)
        self.backbone.to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Hook registration
        self.features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

        # Neighborhood pooling: AvgPool2d(3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1).to(self.device)

        # Memory bank & inference state
        self.memory_bank: Optional[np.ndarray] = None  # (N_coreset, D)
        self.threshold: float = 0.0
        self.calibrated: bool = False
        self.nn_searcher: Optional[NearestNeighbors] = None
        self.timing_profile: Dict[str, Any] = {
            "device": str(self.device),
            "feature_extraction_seconds": 0.0,
            "random_projection_seconds": 0.0,
            "coreset_selection_seconds": 0.0,
            "nearest_neighbors_fit_seconds": 0.0,
            "total_fit_seconds": 0.0,
            "peak_gpu_vram_mib": 0.0,
            "peak_host_ram_mib": 0.0,
        }

    def _load_backbone(self, name: str, weights_path: Optional[str] = None) -> nn.Module:
        """Loads torchvision backbone with pretrained weights."""
        if name == "wide_resnet50_2":
            try:
                from torchvision.models import Wide_ResNet50_2_Weights
                weights = Wide_ResNet50_2_Weights.DEFAULT if weights_path is None else None
                model = models.wide_resnet50_2(weights=weights)
                if weights_path is not None:
                    state = torch.load(weights_path, map_location="cpu")
                    model.load_state_dict(state)
            except Exception:
                model = models.wide_resnet50_2(weights=None)
        elif name == "resnet18":
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if weights_path is None else None
                model = models.resnet18(weights=weights)
                if weights_path is not None:
                    state = torch.load(weights_path, map_location="cpu")
                    model.load_state_dict(state)
            except Exception:
                model = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        return model

    def _register_hooks(self):
        """Attaches forward hooks to extract activations from target layers."""
        def get_hook(name: str):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(get_hook(layer_name))

    def extract_patch_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extracts aligned, aggregated patch embeddings for a batch of images.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            patch_embeddings: Tensor of shape (B * H_patch * W_patch, D)
            patch_grid_shape: (H_patch, W_patch)
        """
        self.features.clear()
        x = x.to(self.device, non_blocking=True)
        with torch.inference_mode():
            _ = self.backbone(x)

        # Extract and pool activations
        layer_feats = []
        base_shape = None

        for idx, layer_name in enumerate(self.layers):
            feat = self.features[layer_name]  # (B, C, H, W)
            feat = self.avg_pool(feat)

            if idx == 0:
                base_shape = feat.shape[2:]  # (H_base, W_base), e.g. (32, 32)
                layer_feats.append(feat)
            else:
                aligned_feat = F.interpolate(
                    feat, size=base_shape, mode="bilinear", align_corners=False
                )
                layer_feats.append(aligned_feat)

        # Clear feature references to release intermediate activations immediately
        self.features.clear()

        # Concatenate along channel dimension: (B, C_total, H_base, W_base)
        concat_feats = torch.cat(layer_feats, dim=1)
        b, c, h, w = concat_feats.shape

        patches = concat_feats.permute(0, 2, 3, 1).reshape(-1, c)
        return patches, (h, w)

    def fit_memory_bank(
        self,
        train_loader: Any,
        verbose: bool = True,
        chunk_size: Optional[int] = None,
        candidate_pool_size: Optional[int] = None,
    ):
        """Constructs normal patch memory bank using memory-bounded greedy k-center.
        
        Optimizations:
        1. Pre-allocated single contiguous float32 raw array (eliminates 1.1 GB duplicate copy)
        2. Chunked random projection
        3. Precomputed squared row norms
        4. Memory-bounded greedy k-center using chunked vector dot products (zero (N, 128) temporary arrays)
        5. Immediate garbage collection of raw features prior to NearestNeighbors fitting
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if candidate_pool_size is not None:
            self.candidate_pool_size = candidate_pool_size

        total_start = time.perf_counter()
        if verbose:
            print(f"[Memory Bank] Initial RSS: {_get_rss_mb():.1f} MiB | Device: {self.device}")

        self.backbone.eval()

        # Step 1: Feature Extraction with single pre-allocated raw memory buffer
        t_feat0 = time.perf_counter()
        # Probe first batch to determine patch dimensions
        first_batch = next(iter(train_loader))
        first_images = first_batch["image"]
        first_patches, grid_shape = self.extract_patch_features(first_images)
        dim = first_patches.shape[1]
        patches_per_image = grid_shape[0] * grid_shape[1]

        total_images = len(train_loader.dataset)
        total_patches = total_images * patches_per_image

        if verbose:
            print(f"[Memory Bank] Normal images: {total_images} | Patches per image: {patches_per_image}")
            print(f"[Memory Bank] Pre-allocating contiguous raw memory: ({total_patches}, {dim}) float32 ({(total_patches * dim * 4) / (1024**2):.1f} MiB)")

        raw_memory = np.empty((total_patches, dim), dtype=np.float32)

        # Copy first batch
        n_first = len(first_patches)
        raw_memory[0:n_first] = first_patches.cpu().numpy()
        cursor = n_first
        del first_patches, first_images, first_batch

        # Fill remaining batches in-place
        for idx, batch in enumerate(train_loader):
            if idx == 0:
                continue  # already processed
            images = batch["image"]
            patches, _ = self.extract_patch_features(images)
            n_p = len(patches)
            raw_memory[cursor : cursor + n_p] = patches.cpu().numpy()
            cursor += n_p
            del patches, images

        feat_time = time.perf_counter() - t_feat0
        self.timing_profile["feature_extraction_seconds"] = round(feat_time, 3)

        if verbose:
            print(f"[Memory Bank] Feature extraction complete in {feat_time:.2f}s: {cursor} patches stored. RSS: {_get_rss_mb():.1f} MiB | VRAM: {_get_vram_mb():.1f} MiB")

        # Step 2: Target coreset size
        target_coreset_size = max(1, int(total_patches * self.coreset_ratio))
        if verbose:
            print(f"[Memory Bank] Coreset ratio: {self.coreset_ratio:.3f} | Target coreset vectors: {target_coreset_size}")

        if target_coreset_size >= total_patches:
            self.memory_bank = raw_memory
        else:
            # Step 3: Chunked random projection to lower dimension
            t_proj0 = time.perf_counter()
            proj_dim = min(self.dimension_reduction, dim)
            rng = np.random.default_rng(42)
            proj_matrix = rng.standard_normal((dim, proj_dim)).astype(np.float32)
            proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)

            if verbose:
                print(f"[Memory Bank] Projecting features to {proj_dim} dimensions in chunks of {self.chunk_size}...")

            features_proj = np.empty((total_patches, proj_dim), dtype=np.float32)
            for p_start in range(0, total_patches, self.chunk_size):
                p_end = min(p_start + self.chunk_size, total_patches)
                features_proj[p_start:p_end] = np.dot(raw_memory[p_start:p_end], proj_matrix)

            del proj_matrix
            proj_time = time.perf_counter() - t_proj0
            self.timing_profile["random_projection_seconds"] = round(proj_time, 3)

            # Step 4: Optional Candidate Pool Subsampling
            effective_features_proj = features_proj
            candidate_indices = None

            if self.candidate_pool_size is not None and self.candidate_pool_size < total_patches:
                if verbose:
                    print(f"[Memory Bank] [APPROXIMATE CORESET] Subsampling candidate pool to {self.candidate_pool_size} patches...")
                candidate_indices = rng.choice(total_patches, size=self.candidate_pool_size, replace=False)
                candidate_indices.sort()
                effective_features_proj = features_proj[candidate_indices]

            # Step 5: Memory-bounded Greedy k-Center Subsampling
            t0 = time.perf_counter()
            sub_indices = self._greedy_kcenter_subsampling(
                effective_features_proj, target_coreset_size, verbose=verbose
            )
            coreset_time = time.perf_counter() - t0
            self.timing_profile["coreset_selection_seconds"] = round(coreset_time, 3)

            if candidate_indices is not None:
                final_selected_indices = candidate_indices[sub_indices]
            else:
                final_selected_indices = sub_indices

            if verbose:
                print(f"[Memory Bank] Coreset selection finished in {coreset_time:.2f}s. Selected: {len(final_selected_indices)} vectors.")

            # Step 6: Retain only the coreset vectors in memory bank
            self.memory_bank = np.ascontiguousarray(raw_memory[final_selected_indices])

            # Release heavy buffers immediately before fitting NearestNeighbors
            del raw_memory
            del features_proj
            if candidate_indices is not None:
                del effective_features_proj
                del candidate_indices
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if verbose:
                print(f"[Memory Bank] Temporary buffers released. RSS: {_get_rss_mb():.1f} MiB")

        # Step 7: Build NearestNeighbors search structure
        t_nn0 = time.perf_counter()
        if verbose:
            print(f"[Memory Bank] Fitting NearestNeighbors on memory bank {self.memory_bank.shape}...")
        self.nn_searcher = NearestNeighbors(
            n_neighbors=self.num_neighbors, algorithm="auto", metric="euclidean", n_jobs=-1
        )
        self.nn_searcher.fit(self.memory_bank)
        nn_time = time.perf_counter() - t_nn0
        self.timing_profile["nearest_neighbors_fit_seconds"] = round(nn_time, 3)

        total_fit = time.perf_counter() - total_start
        self.timing_profile["total_fit_seconds"] = round(total_fit, 3)
        self.timing_profile["peak_gpu_vram_mib"] = round(_get_vram_mb(), 2)
        self.timing_profile["peak_host_ram_mib"] = round(_get_rss_mb(), 2)

        if verbose:
            print(f"[Memory Bank] Ready for inference in {total_fit:.2f}s total. Final RSS: {_get_rss_mb():.1f} MiB | Peak VRAM: {_get_vram_mb():.1f} MiB")

    def _greedy_kcenter_subsampling(
        self, features_proj: np.ndarray, target_size: int, verbose: bool = True
    ) -> np.ndarray:
        """Memory-bounded minimax greedy k-Center subsampling using squared Euclidean distances.
        
        Formulation:
            ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * <x, c>
            
        Guarantees:
        - O(chunk_size) maximum temporary memory allocation (zero (N, D) temporary arrays).
        - Precomputed row-wise squared norms.
        - Selected centers are marked with -1.0 to prevent re-selection.
        - Deterministic execution via fixed RNG seed.
        """
        n_samples, dim = features_proj.shape
        features_proj = np.ascontiguousarray(features_proj, dtype=np.float32)

        # Precompute squared row norms
        feat_sq_norms = np.empty(n_samples, dtype=np.float32)
        for i in range(0, n_samples, self.chunk_size):
            end_i = min(i + self.chunk_size, n_samples)
            feat_sq_norms[i:end_i] = np.sum(features_proj[i:end_i] ** 2, axis=1)

        rng = np.random.default_rng(42)
        selected_indices = np.empty(target_size, dtype=np.int64)

        # Select first center at random
        first_idx = int(rng.integers(0, n_samples))
        selected_indices[0] = first_idx

        # Initialize running minimum squared distances
        min_dist_sq = np.full(n_samples, np.inf, dtype=np.float32)
        first_center = features_proj[first_idx]
        first_c_norm_sq = float(feat_sq_norms[first_idx])

        # Chunked update for first center
        for i in range(0, n_samples, self.chunk_size):
            end_i = min(i + self.chunk_size, n_samples)
            dot_prod = np.dot(features_proj[i:end_i], first_center)
            d_sq = feat_sq_norms[i:end_i] + first_c_norm_sq - 2.0 * dot_prod
            min_dist_sq[i:end_i] = np.maximum(d_sq, 0.0)

        # Prevent first center from being re-selected
        min_dist_sq[first_idx] = -1.0

        step_log = max(1, target_size // 10)
        for step in range(1, target_size):
            new_idx = int(np.argmax(min_dist_sq))
            selected_indices[step] = new_idx

            new_center = features_proj[new_idx]
            new_c_norm_sq = float(feat_sq_norms[new_idx])

            # Chunked update: compute squared distances to new center and take running minimum
            for i in range(0, n_samples, self.chunk_size):
                end_i = min(i + self.chunk_size, n_samples)
                dot_prod = np.dot(features_proj[i:end_i], new_center)
                d_sq = feat_sq_norms[i:end_i] + new_c_norm_sq - 2.0 * dot_prod
                d_sq = np.maximum(d_sq, 0.0)
                min_dist_sq[i:end_i] = np.minimum(min_dist_sq[i:end_i], d_sq)

            # Mask out newly selected center
            min_dist_sq[new_idx] = -1.0

            if verbose and (step % step_log == 0 or step == target_size - 1):
                rss = _get_rss_mb()
                print(f"  Coreset subsampling: {step + 1}/{target_size} selected... [RSS: {rss:.1f} MiB]")

        del feat_sq_norms
        del min_dist_sq

        return selected_indices

    def calibrate_threshold(self, val_loader: Any, target_fpr: float = 0.05) -> float:
        """Calibrates image anomaly threshold on normal validation images for target FPR."""
        val_scores = []
        for batch in val_loader:
            images = batch["image"]
            scores, _ = self.predict_batch(images)
            val_scores.extend(scores.tolist())

        val_scores = np.array(val_scores)
        # (1 - target_fpr) quantile gives ~target_fpr false positive rate on normal images
        threshold = float(np.quantile(val_scores, 1.0 - target_fpr))
        self.threshold = threshold
        self.calibrated = True
        return threshold

    def predict_batch(
        self, images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts image-level anomaly scores and 2D anomaly heatmaps.
        
        Args:
            images: Tensor (B, 3, H, W)
            
        Returns:
            image_scores: np.ndarray (B,)
            anomaly_maps: np.ndarray (B, H_in, W_in)
        """
        if self.nn_searcher is None:
            raise RuntimeError("Model memory bank is not fitted. Fit or load a trained model first.")

        b, _, h_in, w_in = images.shape
        patches, (h_grid, w_grid) = self.extract_patch_features(images)
        patches_np = patches.cpu().numpy()

        # Compute nearest neighbor distances
        distances, _ = self.nn_searcher.kneighbors(patches_np, n_neighbors=1, return_distance=True)
        distances = distances[:, 0]  # (B * h_grid * w_grid,)

        # Reshape to (B, h_grid, w_grid)
        patch_dist_maps = distances.reshape(b, h_grid, w_grid)

        image_scores = []
        anomaly_maps = []

        for i in range(b):
            dist_map = patch_dist_maps[i]

            # Image score = max patch distance
            img_score = float(np.max(dist_map))
            image_scores.append(img_score)

            # Upsample anomaly map to input image resolution (h_in, w_in) using bilinear interpolation
            upsampled = cv2.resize(dist_map, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            # Apply 2D Gaussian filter smoothing (sigma=4) as in PatchCore paper
            smoothed = gaussian_filter(upsampled, sigma=4)
            anomaly_maps.append(smoothed)

        return np.array(image_scores, dtype=np.float32), np.array(anomaly_maps, dtype=np.float32)

    def save(self, save_path: Union[str, Path]):
        """Saves complete model artifact for reproducible inference."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "target_size": self.target_size,
            "coreset_ratio": self.coreset_ratio,
            "num_neighbors": self.num_neighbors,
            "dimension_reduction": self.dimension_reduction,
            "chunk_size": self.chunk_size,
            "candidate_pool_size": self.candidate_pool_size,
            "memory_bank": self.memory_bank,
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "timing_profile": self.timing_profile,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, load_path: Union[str, Path], device: Optional[str] = None) -> "PatchCoreModel":
        """Loads complete trained PatchCore artifact from disk."""
        with open(load_path, "rb") as f:
            state = pickle.load(f)

        model = cls(
            backbone_name=state["backbone_name"],
            layers=state["layers"],
            target_size=state["target_size"],
            coreset_ratio=state["coreset_ratio"],
            num_neighbors=state["num_neighbors"],
            dimension_reduction=state["dimension_reduction"],
            chunk_size=state.get("chunk_size", 8192),
            candidate_pool_size=state.get("candidate_pool_size", None),
            device=device,
        )
        model.memory_bank = state["memory_bank"]
        model.threshold = state["threshold"]
        model.calibrated = state["calibrated"]
        model.timing_profile = state.get("timing_profile", {})

        if model.memory_bank is not None:
            model.nn_searcher = NearestNeighbors(
                n_neighbors=model.num_neighbors, algorithm="auto", metric="euclidean", n_jobs=-1
            )
            model.nn_searcher.fit(model.memory_bank)

        return model
