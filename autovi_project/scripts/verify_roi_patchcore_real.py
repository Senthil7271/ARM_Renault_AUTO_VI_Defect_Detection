"""Real-data CPU verification for ROIPatchCoreModel on pipe_clip."""

import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.roi_patchcore import ROIPatchCoreModel


class RealImageFolderDataset(Dataset):
    def __init__(self, paths: list, target_size=(256, 256)):
        self.paths = paths
        self.target_size = target_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return {"image": t_img, "path": str(p)}


def verify_real_pipeclip():
    train_good_dir = Path("pipe_clip/pipe_clip/train/good")
    all_train_imgs = sorted(list(train_good_dir.glob("*.png")))
    train_subset = all_train_imgs[:15]
    val_subset = all_train_imgs[15:25]

    print("=" * 70)
    print("PHASE 4: REAL-DATA CPU VERIFICATION FOR ROI PATCHCORE")
    print(f"Train subset: {len(train_subset)} normal images")
    print(f"Val subset:   {len(val_subset)} normal images")
    print("=" * 70)

    # 1. Automatic Normal-Only ROI Derivation
    t0 = time.time()
    roi_box = ROIPatchCoreModel.derive_normal_roi_from_paths(train_subset, target_size=(256, 256))
    print(f"[1/5] Derived Normal ROI: ymin={roi_box[0]:.3f}, xmin={roi_box[1]:.3f}, ymax={roi_box[2]:.3f}, xmax={roi_box[3]:.3f} (elapsed: {time.time()-t0:.2f}s)")

    # 2. Initialize Model
    model = ROIPatchCoreModel(
        backbone_name="wide_resnet50_2",
        layers=("layer2", "layer3"),
        target_size=(256, 256),
        coreset_ratio=0.1,
        roi_box=roi_box,
        coord_scale=0.05,
        device="cpu",
    )

    # 3. Fit Memory Bank
    train_ds = RealImageFolderDataset(train_subset, target_size=(256, 256))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)

    print("\n[2/5] Fitting ROI PatchCore Memory Bank...")
    t1 = time.time()
    model.fit_memory_bank(train_loader, verbose=True)
    print(f"Memory bank fitted in {time.time()-t1:.2f}s | Shape: {model.memory_bank.shape}")

    # 4. Calibrate Threshold on Held-Out Nominals Only
    val_ds = RealImageFolderDataset(val_subset, target_size=(256, 256))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print("\n[3/5] Calibrating 95th Percentile Threshold on Normal Validation...")
    threshold = model.calibrate_threshold(val_loader, target_fpr=0.05)
    print(f"Calibrated Threshold: {threshold:.4f} (calibrated: {model.calibrated})")

    # 5. Serialization & Reloading
    out_dir = Path("autovi_project/outputs/phase4/verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "roi_patchcore_test_checkpoint.pt"
    print(f"\n[4/5] Saving checkpoint to: {ckpt_path}...")
    model.save_roi_checkpoint(ckpt_path, metadata={"verified": True, "n_train": len(train_subset)})
    print(f"Saved checkpoint size: {ckpt_path.stat().st_size / (1024*1024):.2f} MB")

    reloaded = ROIPatchCoreModel.load_roi_checkpoint(ckpt_path, device="cpu")
    print(f"Reloaded model threshold: {reloaded.threshold:.4f} (calibrated: {reloaded.calibrated})")

    # 6. Single Image Inference Test
    print("\n[5/5] Running single-image inference test...")
    sample_batch = next(iter(val_loader))
    t_img = sample_batch["image"][:1]
    score, amap = reloaded.predict_anomaly(t_img)
    print(f"Test inference: Score = {score:.4f}, Anomaly Map shape = {amap.shape}, Min = {amap.min():.4f}, Max = {amap.max():.4f}")
    print(f"Score is above threshold: {score >= reloaded.threshold}")

    print("\n[PASS] All Real-Data CPU Integration Checks Succeeded!")


if __name__ == "__main__":
    verify_real_pipeclip()
