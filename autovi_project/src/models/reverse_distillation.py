"""Reverse Distillation (RD) for Industrial Anomaly Detection.

Based on Deng & Li (CVPR 2022) / Bamba et al.
Architecture:
- Frozen Pretrained Teacher: WideResNet-50-2 (or ResNet-50) multi-scale intermediate activations (layer1, layer2, layer3).
- One-Class Embedding (OCE) Bottleneck: Multi-scale feature fusion and compression.
- Trainable Student Decoder: Reverse hierarchical reconstruction of multi-scale teacher representations.
- Objective: Multi-scale Cosine Feature Discrepancy on nominal samples.
- Inference: Multi-scale cosine distance anomaly maps with Top-K / maximum anomaly scoring.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T


class MultiScaleFeatureFusion(nn.Module):
    """Bottleneck One-Class Embedding (OCE) fusing multi-scale teacher features."""

    def __init__(self, in_channels: List[int], bottleneck_dim: int = 512):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim

        # Project each scale to bottleneck_dim // 3
        c_proj = bottleneck_dim // 3
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels[0], c_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_proj),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels[1], c_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_proj),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(in_channels[2], c_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_proj),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Fused convolution
        total_proj = c_proj * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(total_proj, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features[0]: layer1 (B, C1, H/4, W/4)
        # features[1]: layer2 (B, C2, H/8, W/8)
        # features[2]: layer3 (B, C3, H/16, W/16)
        target_size = features[2].shape[-2:]
        p1 = F.adaptive_avg_pool2d(self.proj1(features[0]), target_size)
        p2 = F.adaptive_avg_pool2d(self.proj2(features[1]), target_size)
        p3 = self.proj3(features[2])
        concat = torch.cat([p1, p2, p3], dim=1)
        bottleneck = self.fuse(concat)
        return bottleneck


class StudentDecoderBlock(nn.Module):
    """Upsampling and convolution block for Student Decoder."""

    def __init__(self, in_c: int, out_c: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        layers.extend([
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StudentDecoder(nn.Module):
    """Hierarchical Reverse Distillation Student Decoder."""

    def __init__(self, target_channels: List[int], bottleneck_dim: int = 512):
        super().__init__()
        # target_channels: [C1 (layer1), C2 (layer2), C3 (layer3)]
        c1, c2, c3 = target_channels

        # Stage 3 reconstruction (resolution 16x16)
        self.dec_stage3 = StudentDecoderBlock(bottleneck_dim, c3, upsample=False)

        # Stage 2 reconstruction (resolution 32x32)
        self.dec_stage2 = StudentDecoderBlock(c3, c2, upsample=True)

        # Stage 1 reconstruction (resolution 64x64)
        self.dec_stage1 = StudentDecoderBlock(c2, c1, upsample=True)

    def forward(self, bottleneck: torch.Tensor) -> List[torch.Tensor]:
        f3_hat = self.dec_stage3(bottleneck)
        f2_hat = self.dec_stage2(f3_hat)
        f1_hat = self.dec_stage1(f2_hat)
        return [f1_hat, f2_hat, f3_hat]


class ReverseDistillationModel(nn.Module):
    """Complete Reverse Distillation Anomaly Detection Model."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        bottleneck_dim: int = 512,
        input_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.bottleneck_dim = bottleneck_dim
        self.input_size = input_size
        self.threshold = None
        self.calibrated = False

        # Build Teacher
        self._build_teacher()

        # Build Bottleneck & Student Decoder
        self.oce = MultiScaleFeatureFusion(self.teacher_channels, bottleneck_dim=bottleneck_dim)
        self.decoder = StudentDecoder(self.teacher_channels, bottleneck_dim=bottleneck_dim)

    def _build_teacher(self):
        if self.backbone_name == "wide_resnet50_2":
            local_cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "wide_resnet50_2-9ba9bcbe.pth"
            if local_cached.exists():
                base_model = models.wide_resnet50_2(weights=None)
                state_dict = torch.load(local_cached, map_location="cpu")
                base_model.load_state_dict(state_dict)
            else:
                try:
                    weights = models.Wide_ResNet50_2_Weights.DEFAULT
                    base_model = models.wide_resnet50_2(weights=weights)
                except Exception:
                    base_model = models.wide_resnet50_2(weights=None)
            self.teacher_channels = [256, 512, 1024]
        elif self.backbone_name == "resnet18":
            local_cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
            if local_cached.exists():
                base_model = models.resnet18(weights=None)
                state_dict = torch.load(local_cached, map_location="cpu")
                base_model.load_state_dict(state_dict)
            else:
                try:
                    weights = models.ResNet18_Weights.DEFAULT
                    base_model = models.resnet18(weights=weights)
                except Exception:
                    base_model = models.resnet18(weights=None)
            self.teacher_channels = [64, 128, 256]
        elif self.backbone_name == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT
                base_model = models.resnet50(weights=weights)
            except Exception:
                base_model = models.resnet50(weights=None)
            self.teacher_channels = [256, 512, 1024]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        self.teacher_conv1 = base_model.conv1
        self.teacher_bn1 = base_model.bn1
        self.teacher_relu = base_model.relu
        self.teacher_maxpool = base_model.maxpool
        self.teacher_layer1 = base_model.layer1
        self.teacher_layer2 = base_model.layer2
        self.teacher_layer3 = base_model.layer3

        # Strictly freeze teacher
        for p in self.teacher_parameters():
            p.requires_grad = False

    def teacher_parameters(self):
        for m in [
            self.teacher_conv1,
            self.teacher_bn1,
            self.teacher_relu,
            self.teacher_maxpool,
            self.teacher_layer1,
            self.teacher_layer2,
            self.teacher_layer3,
        ]:
            for p in m.parameters():
                yield p

    def trainable_parameters(self):
        for p in self.oce.parameters():
            if p.requires_grad:
                yield p
        for p in self.decoder.parameters():
            if p.requires_grad:
                yield p

    def extract_teacher_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            x = self.teacher_conv1(x)
            x = self.teacher_bn1(x)
            x = self.teacher_relu(x)
            x = self.teacher_maxpool(x)
            f1 = self.teacher_layer1(x)
            f2 = self.teacher_layer2(f1)
            f3 = self.teacher_layer3(f2)
        return [f1, f2, f3]

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass during training: returns teacher features and student reconstructed features."""
        # Teacher is evaluated in eval mode (no grad)
        teacher_feats = self.extract_teacher_features(x)
        # Trainable Student pathway
        bottleneck = self.oce(teacher_feats)
        student_feats = self.decoder(bottleneck)
        return teacher_feats, student_feats

    @staticmethod
    def compute_cosine_discrepancy_loss(
        teacher_feats: List[torch.Tensor],
        student_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """Computes multi-scale cosine distance loss between normalized teacher and student features."""
        total_loss = torch.tensor(0.0, device=teacher_feats[0].device)
        for ft, fs in zip(teacher_feats, student_feats):
            # Normalize along channel dimension
            ft_norm = F.normalize(ft, p=2, dim=1)
            fs_norm = F.normalize(fs, p=2, dim=1)
            # Cosine similarity per spatial position: (B, H, W)
            cos_sim = torch.sum(ft_norm * fs_norm, dim=1)
            # Cosine distance = 1 - cos_sim
            scale_loss = torch.mean(1.0 - cos_sim)
            total_loss = total_loss + scale_loss
        return total_loss

    def predict_anomaly(
        self,
        x: torch.Tensor,
        top_k_ratio: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference: generates pixel-level anomaly maps and image-level anomaly scores.
        
        Args:
            x: Input tensor (B, 3, H, W).
            top_k_ratio: fraction of top spatial pixels to average for image-level score.
            
        Returns:
            image_scores: Tensor (B,).
            anomaly_maps: Tensor (B, H, W).
        """
        self.eval()
        with torch.no_grad():
            t_feats, s_feats = self.forward(x)
            b, _, h_in, w_in = x.shape
            accum_dist_map = torch.zeros((b, 1, h_in, w_in), device=x.device)

            for ft, fs in zip(t_feats, s_feats):
                ft_norm = F.normalize(ft, p=2, dim=1)
                fs_norm = F.normalize(fs, p=2, dim=1)
                cos_dist = 1.0 - torch.sum(ft_norm * fs_norm, dim=1, keepdim=True)  # (B, 1, H_k, W_k)
                # Bilinearly upsample to input resolution
                dist_up = F.interpolate(cos_dist, size=(h_in, w_in), mode="bilinear", align_corners=True)
                accum_dist_map = accum_dist_map + dist_up

            # Average across scales
            anomaly_map = accum_dist_map / len(t_feats)
            anomaly_map_sq = anomaly_map.squeeze(1)  # (B, H, W)

            # Compute image scores: Top-K pixel mean
            b, h, w = anomaly_map_sq.shape
            flat_maps = anomaly_map_sq.view(b, h * w)
            k = max(1, int(h * w * top_k_ratio))
            top_k_vals, _ = torch.topk(flat_maps, k=k, dim=1)
            image_scores = torch.mean(top_k_vals, dim=1)

            return image_scores, anomaly_map_sq

    def save(self, path: Union[str, Path], metadata: Optional[Dict] = None):
        """Saves model weights, architecture hyperparameters, and calibration metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_type": "reverse_distillation",
            "backbone_name": self.backbone_name,
            "bottleneck_dim": self.bottleneck_dim,
            "input_size": self.input_size,
            "oce_state_dict": self.oce.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "metadata": metadata or {},
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "ReverseDistillationModel":
        """Loads a saved Reverse Distillation model."""
        state = torch.load(path, map_location=device)
        model = cls(
            backbone_name=state["backbone_name"],
            bottleneck_dim=state["bottleneck_dim"],
            input_size=tuple(state["input_size"]),
        )
        model.oce.load_state_dict(state["oce_state_dict"])
        model.decoder.load_state_dict(state["decoder_state_dict"])
        model.threshold = state.get("threshold", None)
        model.calibrated = state.get("calibrated", False)
        model.to(device)
        model.eval()
        return model
