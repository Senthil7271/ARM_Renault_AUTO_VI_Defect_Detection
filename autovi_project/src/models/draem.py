"""DRAEM-style Reconstructive and Discriminative Network for Synthetic Anomaly Learning.

Based on Zavrtanik et al. (ICCV 2021): "DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection"
Architecture:
- Reconstructive Subnetwork (G_R): Encoder-decoder learning to reconstruct nominal images from pseudo-anomalies.
- Discriminative Subnetwork (G_D): U-Net taking concatenated [I_a, I_r] (6 channels) to segment anomalies.
- Training: Supervised entirely on procedural synthetic anomalies generated on-the-fly from nominal training images.
- Inference: Pixel-level anomaly localization map and Top-K image-level anomaly score.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Calculates differentiable 1 - SSIM between two RGB images in range [0, 1]."""
    c = img1.size(1)
    # 1D Gaussian kernel
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    # 2D Gaussian kernel
    window = g.outer(g).unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=c) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return torch.clamp(1.0 - ssim_map.mean(), 0.0, 2.0)


def focal_dice_loss(pred_probs: torch.Tensor, target_masks: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Combines Focal Loss and Soft Dice Loss for imbalanced anomaly segmentation."""
    eps = 1e-6
    p = torch.clamp(pred_probs, eps, 1.0 - eps)
    t = target_masks

    # Focal loss
    pt = torch.where(t == 1, p, 1.0 - p)
    at = torch.where(t == 1, alpha, 1.0 - alpha)
    focal = -at * ((1.0 - pt) ** gamma) * torch.log(pt)
    focal_loss = focal.mean()

    # Dice loss
    intersection = 2.0 * torch.sum(p * t, dim=(1, 2, 3)) + eps
    union = torch.sum(p, dim=(1, 2, 3)) + torch.sum(t, dim=(1, 2, 3)) + eps
    dice_loss = (1.0 - intersection / union).mean()

    return focal_loss + dice_loss


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReconstructiveUNet(nn.Module):
    """Reconstruction Subnetwork learning to restore nominal images from corrupted inputs."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_c: int = 48):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_c * 8, base_c * 8)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = ConvBlock(base_c * 8 + base_c * 8, base_c * 4)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ConvBlock(base_c * 4 + base_c * 4, base_c * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(base_c * 2 + base_c * 2, base_c)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(base_c + base_c, base_c)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_c, out_channels, kernel_size=1),
            nn.Sigmoid(),  # Bound output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


class DiscriminativeUNet(nn.Module):
    """Discriminative Localization Subnetwork predicting pixel anomaly masks."""

    def __init__(self, in_channels: int = 6, out_channels: int = 1, base_c: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_c * 8, base_c * 8)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = ConvBlock(base_c * 8 + base_c * 8, base_c * 4)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ConvBlock(base_c * 4 + base_c * 4, base_c * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(base_c * 2 + base_c * 2, base_c)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(base_c + base_c, base_c)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_c, out_channels, kernel_size=1),
            nn.Sigmoid(),  # Probability map [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


class DRAEMModel(nn.Module):
    """Complete DRAEM Architecture combining Reconstruction and Discrimination."""

    def __init__(
        self,
        base_reconstruct_c: int = 48,
        base_discriminative_c: int = 32,
        input_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.base_reconstruct_c = base_reconstruct_c
        self.base_discriminative_c = base_discriminative_c
        self.input_size = input_size
        self.reconstructor = ReconstructiveUNet(in_channels=3, out_channels=3, base_c=base_reconstruct_c)
        self.discriminator = DiscriminativeUNet(in_channels=6, out_channels=1, base_c=base_discriminative_c)
        self.threshold = None
        self.calibrated = False

    def forward(self, corrupted_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns reconstructed images and predicted anomaly maps."""
        reconstructed = self.reconstructor(corrupted_imgs)
        # Concatenate corrupted input and reconstruction along channels (6 channels)
        disc_input = torch.cat([corrupted_imgs, reconstructed], dim=1)
        anomaly_map = self.discriminator(disc_input)
        return reconstructed, anomaly_map

    def compute_total_loss(
        self,
        nominal_clean: torch.Tensor,
        corrupted_input: torch.Tensor,
        synth_masks: torch.Tensor,
        l2_weight: float = 1.0,
        ssim_weight: float = 2.0,
        disc_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes combined reconstructive (L2 + SSIM) and discriminative (Focal + Dice) loss."""
        reconstructed, pred_anomaly_maps = self.forward(corrupted_input)

        # 1. Reconstruction Loss
        loss_l2 = F.mse_loss(reconstructed, nominal_clean)
        loss_ssim = ssim_loss(reconstructed, nominal_clean)
        loss_rec = l2_weight * loss_l2 + ssim_weight * loss_ssim

        # 2. Discriminative Loss
        loss_disc = focal_dice_loss(pred_anomaly_maps, synth_masks)

        # Total Loss
        total_loss = loss_rec + disc_weight * loss_disc

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_rec": float(loss_rec.item()),
            "loss_l2": float(loss_l2.item()),
            "loss_ssim": float(loss_ssim.item()),
            "loss_disc": float(loss_disc.item()),
        }
        return total_loss, metrics

    def predict_anomaly(
        self,
        x: torch.Tensor,
        top_k_ratio: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference: predicts pixel-level anomaly maps and image-level scores.
        
        Args:
            x: Input tensor (B, 3, H, W) in [0, 1].
            top_k_ratio: Top fraction of pixels to average for image-level score.
            
        Returns:
            image_scores: Tensor (B,).
            anomaly_maps: Tensor (B, H, W).
        """
        self.eval()
        with torch.no_grad():
            _, anomaly_map = self.forward(x)  # (B, 1, H, W)
            anomaly_map_sq = anomaly_map.squeeze(1)  # (B, H, W)

            b, h, w = anomaly_map_sq.shape
            flat_maps = anomaly_map_sq.view(b, h * w)
            k = max(1, int(h * w * top_k_ratio))
            top_k_vals, _ = torch.topk(flat_maps, k=k, dim=1)
            image_scores = torch.mean(top_k_vals, dim=1)

            return image_scores, anomaly_map_sq

    def save(self, path: Union[str, Path], metadata: Optional[Dict] = None):
        """Saves model weights, architecture parameters, and calibration metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_type": "draem",
            "base_reconstruct_c": self.base_reconstruct_c,
            "base_discriminative_c": self.base_discriminative_c,
            "input_size": self.input_size,
            "reconstructor_state_dict": self.reconstructor.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "metadata": metadata or {},
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "DRAEMModel":
        """Loads a saved DRAEM model."""
        state = torch.load(path, map_location=device)
        model = cls(
            base_reconstruct_c=state.get("base_reconstruct_c", 48),
            base_discriminative_c=state.get("base_discriminative_c", 32),
            input_size=tuple(state["input_size"]),
        )
        model.reconstructor.load_state_dict(state["reconstructor_state_dict"])
        model.discriminator.load_state_dict(state["discriminator_state_dict"])
        model.threshold = state.get("threshold", None)
        model.calibrated = state.get("calibrated", False)
        model.to(device)
        model.eval()
        return model
