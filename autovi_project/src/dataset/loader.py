"""AutoVI Dataset and DataLoader implementation.

Implements:
- Strict normal-only training loader
- Reproducible deterministic train/validation split (90/10) with manifest tracking
- Official untouched test loader
- Ground truth mask association with union combination
- Standard ImageNet preprocessing and deterministic resizing
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

from .masks import MaskProcessor


class AutoVIDataset(Dataset):
    """PyTorch Dataset for AutoVI industrial anomaly detection."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        category_root: Path,
        target_size: Tuple[int, int] = (256, 256),
        is_test: bool = False,
    ):
        """
        Args:
            samples: List of sample dicts:
                     {"image_path": Path, "defect_type": str, "is_anomaly": int}
            category_root: Path to resolved category root directory
            target_size: Target (H, W) for image resizing
            is_test: Whether this is the test split (requires mask loading)
        """
        self.samples = samples
        self.category_root = Path(category_root)
        self.target_size = target_size
        self.is_test = is_test
        self.mask_processor = MaskProcessor(target_size=target_size)

        # Standard ImageNet normalization for pretrained CNN backbones
        self.transform = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        defect_type = sample["defect_type"]
        is_anomaly = sample["is_anomaly"]

        # Read image with PIL for torchvision transform
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size
        img_tensor = self.transform(pil_img)

        # Handle mask
        if self.is_test and is_anomaly:
            mask_paths = self.mask_processor.find_mask_paths(
                self.category_root, defect_type, img_path.stem
            )
            binary_mask, multiclass_mask = self.mask_processor.load_combined_mask(
                mask_paths, original_shape=(orig_h, orig_w)
            )
        else:
            binary_mask, multiclass_mask = self.mask_processor.load_combined_mask(
                [], original_shape=(orig_h, orig_w)
            )

        mask_tensor = torch.from_numpy(binary_mask).long()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": is_anomaly,
            "image_path": str(img_path),
            "defect_type": defect_type,
            "original_height": orig_h,
            "original_width": orig_w,
        }


def create_autovi_dataloaders(
    workspace_root: Union[str, Path],
    category: str,
    target_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    train_val_ratio: float = 0.9,
    seed: int = 42,
    num_workers: int = 0,
    manifest_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Creates train, validation, and test dataloaders for AutoVI.
    
    Returns:
        (train_loader, val_loader, test_loader, meta_info)
    """
    root = Path(workspace_root).resolve()
    cat_dir = root / category
    if (cat_dir / category).is_dir():
        cat_root = cat_dir / category
    else:
        cat_root = cat_dir

    train_good_dir = cat_root / "train" / "good"
    all_train_paths = sorted(list(train_good_dir.glob("*.png")) + list(train_good_dir.glob("*.jpg")))

    # Deterministic train/val split
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_train_paths))
    rng.shuffle(indices)

    split_idx = int(len(all_train_paths) * train_val_ratio)
    train_indices = sorted(indices[:split_idx].tolist())
    val_indices = sorted(indices[split_idx:].tolist())

    train_samples = [
        {"image_path": all_train_paths[i], "defect_type": "good", "is_anomaly": 0}
        for i in train_indices
    ]
    val_samples = [
        {"image_path": all_train_paths[i], "defect_type": "good", "is_anomaly": 0}
        for i in val_indices
    ]

    # Save manifest if requested
    if manifest_dir is not None:
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "category": category,
            "seed": seed,
            "train_val_ratio": train_val_ratio,
            "train_images": [str(s["image_path"].name) for s in train_samples],
            "val_images": [str(s["image_path"].name) for s in val_samples],
        }
        with open(manifest_dir / f"{category}_split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # Official Test split
    test_dir = cat_root / "test"
    test_samples = []
    for sub in sorted(list(test_dir.iterdir())):
        if not sub.is_dir():
            continue
        sub_name = sub.name
        is_anom = 0 if sub_name == "good" else 1
        sub_imgs = sorted(list(sub.glob("*.png")) + list(sub.glob("*.jpg")))
        for p in sub_imgs:
            test_samples.append({
                "image_path": p,
                "defect_type": sub_name,
                "is_anomaly": is_anom,
            })

    train_ds = AutoVIDataset(train_samples, cat_root, target_size=target_size, is_test=False)
    val_ds = AutoVIDataset(val_samples, cat_root, target_size=target_size, is_test=False)
    test_ds = AutoVIDataset(test_samples, cat_root, target_size=target_size, is_test=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )

    meta = {
        "category": category,
        "category_root": str(cat_root),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "num_test_good": sum(1 for s in test_samples if s["is_anomaly"] == 0),
        "num_test_defective": sum(1 for s in test_samples if s["is_anomaly"] == 1),
    }

    return train_loader, val_loader, test_loader, meta
