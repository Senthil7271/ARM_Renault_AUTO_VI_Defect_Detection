"""Synthetic Anomaly Generation for AutoVI Industrial Anomaly Detection (DRAEM-style).

Scientific principles:
1. Purely unsupervised / self-supervised generation using only normal training images.
2. Procedural masking: Perlin noise, random polygons, and structured rectangles.
3. Multi-source anomaly textures:
   - Cut-and-paste patches from another random normal training image (component misplacement / foreign object).
   - Local geometric displacement (simulating shifted/loose retaining clips and displaced wiring).
   - Synthetic missing-region corruption (filling mask with local background / inpainting, simulating missing screws or clips).
   - Controlled texture perturbation (frequency-modulated Gaussian noise, color jitter).
   - Specular / lighting flare perturbations (simulating reflective variations).
4. Strict safety constraints:
   - NO real defect test images or test masks are ever accessed or referenced.
   - Ground truth masks M in [0, 1] are generated synthetically alongside the perturbed image.
"""

import math
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def generate_perlin_noise_2d(shape: Tuple[int, int], res: Tuple[int, int]) -> np.ndarray:
    """Generates 2D procedural Perlin-like continuous smooth noise for any arbitrary shape."""
    h, w = shape
    res_h, res_w = max(2, res[0]), max(2, res[1])
    
    # Generate random gradient grid
    low_h, low_w = res_h, res_w
    raw_noise = np.random.randn(low_h, low_w).astype(np.float32)
    
    # Smooth with Gaussian blur
    raw_noise = cv2.GaussianBlur(raw_noise, (3, 3), sigmaX=1.0)
    
    # Upsample to full resolution with bicubic interpolation
    upsampled = cv2.resize(raw_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1]
    noise_min = float(upsampled.min())
    noise_max = float(upsampled.max())
    if noise_max - noise_min > 1e-6:
        noise = (upsampled - noise_min) / (noise_max - noise_min)
    else:
        noise = np.zeros((h, w), dtype=np.float32)
        
    return noise.astype(np.float32)


def generate_synthetic_mask(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generates a binary pseudo-anomaly mask M in {0, 1}^{H, W}."""
    if rng is None:
        rng = np.random.default_rng()
        
    mask_type = rng.choice(["perlin", "polygon", "ellipse", "stripes"])
    mask = np.zeros((height, width), dtype=np.float32)
    
    if mask_type == "perlin":
        res_candidates = [(2, 2), (4, 4), (8, 8)]
        res = res_candidates[rng.integers(0, len(res_candidates))]
        noise = generate_perlin_noise_2d((height, width), res)
        # Random threshold to select 2% to 15% area
        thresh = rng.uniform(0.70, 0.90)
        mask = (noise > thresh).astype(np.float32)
        # Apply morphology to smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
    elif mask_type == "polygon":
        # Random polygon representing misplaced or deformed clip / bracket
        n_points = rng.integers(3, 8)
        cx, cy = rng.integers(int(width * 0.15), int(width * 0.85)), rng.integers(int(height * 0.15), int(height * 0.85))
        radius = rng.integers(int(min(height, width) * 0.05), int(min(height, width) * 0.20))
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_points))
        points = []
        for angle in angles:
            r = radius * rng.uniform(0.6, 1.4)
            px = int(np.clip(cx + r * np.cos(angle), 0, width - 1))
            py = int(np.clip(cy + r * np.sin(angle), 0, height - 1))
            points.append([px, py])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1.0)
        
    elif mask_type == "ellipse":
        # Missing screw / hole or fastener anomaly
        cx = int(rng.integers(int(width * 0.15), int(width * 0.85)))
        cy = int(rng.integers(int(height * 0.15), int(height * 0.85)))
        axes = (int(rng.integers(10, 35)), int(rng.integers(10, 35)))
        angle = float(rng.uniform(0, 360))
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 1.0, -1)
        
    else:  # stripes (e.g. misrouted wire or displaced hose)
        pt1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        pt2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        thickness = int(rng.integers(4, 16))
        cv2.line(mask, pt1, pt2, 1.0, thickness)
        
    # Gaussian blur the mask boundary slightly for realistic smooth blending
    mask_blurred = cv2.GaussianBlur(mask, (7, 7), sigmaX=1.5)
    mask_binary = (mask_blurred > 0.2).astype(np.float32)
    
    # If mask is empty due to edge clipping, fall back to center circle
    if np.sum(mask_binary) < 20:
        cv2.circle(mask_binary, (width // 2, height // 2), 20, 1.0, -1)
        
    return mask_binary


class SyntheticAnomalyGenerator:
    """Self-supervised procedural pseudo-anomaly generator for industrial imagery.
    
    Supports:
    1. Cut-and-paste from another nominal image in the training pool.
    2. Local geometric displacement (shifted retaining clips / misaligned components).
    3. Missing-region inpainting / deletion (simulating missing screws, clips).
    4. Controlled texture & frequency noise perturbation.
    5. Specular / lighting flare simulation.
    """
    
    def __init__(
        self,
        nominal_pool: Optional[List[np.ndarray]] = None,
        p_anomaly: float = 0.5,
        seed: int = 42,
    ):
        self.nominal_pool = nominal_pool or []
        self.p_anomaly = p_anomaly
        self.rng = np.random.default_rng(seed)
        
    def set_nominal_pool(self, pool: List[np.ndarray]):
        self.nominal_pool = pool
        
    def generate(
        self,
        image_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Applies a synthetic anomaly transformation to a nominal image.
        
        Args:
            image_np: RGB float32 image in range [0, 1], shape (H, W, 3).
            
        Returns:
            anomalous_image: RGB float32 in [0, 1], shape (H, W, 3).
            mask: float32 in [0, 1], shape (H, W).
            is_anomaly: 1 if anomaly injected, 0 if clean nominal.
        """
        h, w, c = image_np.shape
        
        # Decide whether to inject anomaly
        if self.rng.uniform(0, 1) > self.p_anomaly:
            return image_np.copy(), np.zeros((h, w), dtype=np.float32), 0
            
        mask = generate_synthetic_mask(h, w, rng=self.rng)
        mask_3d = np.expand_dims(mask, axis=-1)  # (H, W, 1)
        
        transform_type = self.rng.choice([
            "cut_paste_donor",
            "geometric_displacement",
            "missing_region_deletion",
            "texture_noise",
            "specular_reflection",
        ])
        
        if transform_type == "cut_paste_donor" and len(self.nominal_pool) > 1:
            # Sample donor image from normal pool
            donor_idx = self.rng.integers(0, len(self.nominal_pool))
            donor = self.nominal_pool[donor_idx]
            # Random slight rotation or scaling on donor
            alpha = self.rng.uniform(0.6, 1.0)
            anomaly_region = donor * alpha + image_np * (1.0 - alpha)
            perturbed = image_np * (1.0 - mask_3d) + anomaly_region * mask_3d
            
        elif transform_type == "geometric_displacement":
            # Shift the masked region spatially by dx, dy
            dx = int(self.rng.integers(-25, 25))
            dy = int(self.rng.integers(-25, 25))
            m_trans = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_img = cv2.warpAffine(image_np, m_trans, (w, h), borderMode=cv2.BORDER_REFLECT)
            alpha = self.rng.uniform(0.7, 1.0)
            perturbed = image_np * (1.0 - mask_3d) + (shifted_img * alpha + image_np * (1.0 - alpha)) * mask_3d
            
        elif transform_type == "missing_region_deletion":
            # Inpainting / blurring the masked area (simulating missing screw, clip, or fastener)
            blurred = cv2.GaussianBlur(image_np, (25, 25), sigmaX=10.0)
            # Add subtle dark tint to simulate hollow cavity or missing part
            cavity = blurred * self.rng.uniform(0.3, 0.7)
            perturbed = image_np * (1.0 - mask_3d) + cavity * mask_3d
            
        elif transform_type == "texture_noise":
            # Frequency-modulated noise
            freq = self.rng.integers(4, 16)
            noise_2d = generate_perlin_noise_2d((h, w), (freq, freq))
            noise_3d = np.repeat(np.expand_dims(noise_2d, -1), 3, axis=-1)
            # Color perturbation
            color_shift = self.rng.uniform(-0.3, 0.3, size=(1, 1, 3))
            noisy_patch = np.clip(image_np * (0.6 + 0.8 * noise_3d) + color_shift, 0.0, 1.0)
            alpha = self.rng.uniform(0.5, 0.9)
            perturbed = image_np * (1.0 - mask_3d) + (noisy_patch * alpha + image_np * (1.0 - alpha)) * mask_3d
            
        else:  # specular_reflection
            # High intensity flare localized inside mask
            flare = np.clip(image_np + self.rng.uniform(0.4, 0.8), 0.0, 1.0)
            alpha = self.rng.uniform(0.6, 0.95)
            perturbed = image_np * (1.0 - mask_3d) + (flare * alpha + image_np * (1.0 - alpha)) * mask_3d
            
        perturbed = np.clip(perturbed, 0.0, 1.0).astype(np.float32)
        return perturbed, mask, 1
