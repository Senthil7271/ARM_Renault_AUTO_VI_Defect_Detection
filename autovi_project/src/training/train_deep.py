"""Trainable Deep Anomaly Detection Training Engine (Reverse Distillation & DRAEM).

Key engineering features:
1. Strict normal-only training data protocol (Zero test labels, zero test defects).
2. DataLoaders with mild non-destructive geometric/lighting augmentations.
3. AdamW optimizer with Cosine Annealing learning rate schedule.
4. Validation-loss-driven checkpoint selection (best validation loss on held-out nominals).
5. Gradient norm tracking, numerical stability checks, and NaN detection.
6. CUDA mixed precision (torch.cuda.amp) on GPU, robust float32 on CPU.
7. Model serialization with complete reproducibility metadata.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

from src.dataset.synthetic_anomalies import SyntheticAnomalyGenerator
from src.models.draem import DRAEMModel
from src.models.reverse_distillation import ReverseDistillationModel


class NominalImageDataset(Dataset):
    """Dataset for Reverse Distillation loading nominal images with normalization."""

    def __init__(
        self,
        image_paths: List[Path],
        target_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
    ):
        self.image_paths = image_paths
        self.target_size = target_size
        self.is_train = is_train

        # Standard ImageNet transforms for pretrained teacher
        if is_train:
            self.transform = T.Compose([
                T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.05, contrast=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class SyntheticAnomalyDataset(Dataset):
    """Dataset for DRAEM generating on-the-fly pseudo-anomalies from nominal training images."""

    def __init__(
        self,
        image_paths: List[Path],
        target_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
        p_anomaly: float = 0.5,
        seed: int = 42,
    ):
        self.image_paths = image_paths
        self.target_size = target_size
        self.is_train = is_train
        self.p_anomaly = p_anomaly

        # Pre-cache nominal images in numpy [0, 1] for fast donor sampling
        self.cached_nominals = []
        for p in image_paths[:50]:  # Cache up to 50 for memory safety
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                self.cached_nominals.append(img.astype(np.float32) / 255.0)

        self.generator = SyntheticAnomalyGenerator(
            nominal_pool=self.cached_nominals,
            p_anomaly=p_anomaly if is_train else 0.5,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_np = (img.astype(np.float32) / 255.0)

        # Generate pseudo-anomaly
        corrupted_np, mask_np, _ = self.generator.generate(img_np)

        # Convert to float32 tensors: Clean (3, H, W), Corrupted (3, H, W), Mask (1, H, W)
        t_clean = torch.from_numpy(img_np).permute(2, 0, 1).float()
        t_corrupted = torch.from_numpy(corrupted_np).permute(2, 0, 1).float()
        t_mask = torch.from_numpy(mask_np).unsqueeze(0).float()

        return t_clean, t_corrupted, t_mask


def train_reverse_distillation(
    train_paths: List[Path],
    val_paths: List[Path],
    output_dir: Path,
    backbone_name: str = "wide_resnet50_2",
    bottleneck_dim: int = 512,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_amp: bool = True,
    seed: int = 42,
    patience: int = 10,
) -> Tuple[ReverseDistillationModel, Dict]:
    """Trains Reverse Distillation on nominal images."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets & Loaders
    train_ds = NominalImageDataset(train_paths, is_train=True)
    val_ds = NominalImageDataset(val_paths, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Model & Optimizer
    model = ReverseDistillationModel(backbone_name=backbone_name, bottleneck_dim=bottleneck_dim)
    model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    teacher_params = sum(p.numel() for p in model.teacher_parameters())
    print(f"[RD] Initialized model: {trainable_params:,} trainable params, {teacher_params:,} frozen teacher params.")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    history = {
        "train_loss": [],
        "val_loss": [],
        "grad_norm": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_epoch = 0.0
        grad_norm_epoch = 0.0
        batches = 0

        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                t_feats, s_feats = model(images)
                loss = model.compute_cosine_discrepancy_loss(t_feats, s_feats)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected at epoch {epoch}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_epoch += loss.item()
            grad_norm_epoch += grad_norm.item()
            batches += 1

        scheduler.step()
        train_loss_avg = train_loss_epoch / max(1, batches)
        grad_norm_avg = grad_norm_epoch / max(1, batches)

        # Validation on held-out nominals
        model.eval()
        val_loss_epoch = 0.0
        val_batches = 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                t_feats, s_feats = model(images)
                val_loss = model.compute_cosine_discrepancy_loss(t_feats, s_feats)
                val_loss_epoch += val_loss.item()
                val_batches += 1
        val_loss_avg = val_loss_epoch / max(1, val_batches)

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["grad_norm"].append(grad_norm_avg)
        history["learning_rates"].append(scheduler.get_last_lr()[0])

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[RD Epoch {epoch:02d}/{epochs:02d}] Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Grad Norm: {grad_norm_avg:.3f}")

        # Checkpoint selection: best validation loss
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            model.save(output_dir / "best_model.pth", metadata={"epoch": epoch, "val_loss": val_loss_avg})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[RD] Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    # Save final model and history
    model.save(output_dir / "final_model.pth", metadata={"epochs": epoch, "final_val_loss": val_loss_avg})
    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    summary = {
        "architecture": "reverse_distillation",
        "backbone": backbone_name,
        "trainable_params": trainable_params,
        "frozen_params": teacher_params,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "training_time_sec": total_time,
    }
    return model, summary


def train_draem(
    train_paths: List[Path],
    val_paths: List[Path],
    output_dir: Path,
    base_reconstruct_c: int = 48,
    base_discriminative_c: int = 32,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_amp: bool = True,
    seed: int = 42,
    patience: int = 10,
) -> Tuple[DRAEMModel, Dict]:
    """Trains DRAEM on nominal images with self-supervised pseudo-anomalies."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets & Loaders
    train_ds = SyntheticAnomalyDataset(train_paths, is_train=True, seed=seed)
    val_ds = SyntheticAnomalyDataset(val_paths, is_train=False, seed=seed + 100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Model & Optimizer
    model = DRAEMModel(base_reconstruct_c=base_reconstruct_c, base_discriminative_c=base_discriminative_c)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DRAEM] Initialized model: {trainable_params:,} trainable params.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    history = {
        "train_loss": [],
        "val_loss": [],
        "loss_rec": [],
        "loss_disc": [],
        "grad_norm": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_epoch = 0.0
        rec_loss_epoch = 0.0
        disc_loss_epoch = 0.0
        grad_norm_epoch = 0.0
        batches = 0

        for clean, corrupted, mask in train_loader:
            clean = clean.to(device)
            corrupted = corrupted.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                loss, metrics = model.compute_total_loss(clean, corrupted, mask)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected at epoch {epoch}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_epoch += loss.item()
            rec_loss_epoch += metrics["loss_rec"]
            disc_loss_epoch += metrics["loss_disc"]
            grad_norm_epoch += grad_norm.item()
            batches += 1

        scheduler.step()
        train_loss_avg = train_loss_epoch / max(1, batches)
        rec_loss_avg = rec_loss_epoch / max(1, batches)
        disc_loss_avg = disc_loss_epoch / max(1, batches)
        grad_norm_avg = grad_norm_epoch / max(1, batches)

        # Validation on held-out nominals
        model.eval()
        val_loss_epoch = 0.0
        val_batches = 0
        with torch.no_grad():
            for clean, corrupted, mask in val_loader:
                clean = clean.to(device)
                corrupted = corrupted.to(device)
                mask = mask.to(device)
                val_loss, _ = model.compute_total_loss(clean, corrupted, mask)
                val_loss_epoch += val_loss.item()
                val_batches += 1
        val_loss_avg = val_loss_epoch / max(1, val_batches)

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["loss_rec"].append(rec_loss_avg)
        history["loss_disc"].append(disc_loss_avg)
        history["grad_norm"].append(grad_norm_avg)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[DRAEM Epoch {epoch:02d}/{epochs:02d}] Total: {train_loss_avg:.4f} (Rec: {rec_loss_avg:.4f}, Disc: {disc_loss_avg:.4f}) | Val: {val_loss_avg:.4f} | Grad: {grad_norm_avg:.3f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            model.save(output_dir / "best_model.pth", metadata={"epoch": epoch, "val_loss": val_loss_avg})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[DRAEM] Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    model.save(output_dir / "final_model.pth", metadata={"epochs": epoch, "final_val_loss": val_loss_avg})
    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    summary = {
        "architecture": "draem",
        "trainable_params": trainable_params,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "training_time_sec": total_time,
    }
    return model, summary
