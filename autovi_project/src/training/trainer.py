"""PatchCore Trainer Module.

Orchestrates:
- Normal-only memory bank construction
- Reproducible validation calibration
- Model serialization & reload verification
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from ..dataset.loader import create_autovi_dataloaders
from ..models.patchcore_model import PatchCoreModel


class PatchCoreTrainer:
    """Trains and calibrates a PatchCore model on normal industrial images."""

    def __init__(
        self,
        workspace_root: Union[str, Path],
        category: str,
        config: Dict[str, Any],
        output_dir: Union[str, Path],
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.category = category
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = self.output_dir / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> PatchCoreModel:
        """Executes normal-only training and threshold calibration."""
        print(f"\n=======================================================")
        print(f"Starting PatchCore Training for Category: {self.category}")
        print(f"=======================================================")

        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        target_size = tuple(data_cfg.get("image_size", [256, 256]))
        batch_size = data_cfg.get("batch_size", 8)
        num_workers = data_cfg.get("num_workers", 0)
        seed = self.config.get("project", {}).get("seed", 42)
        train_val_ratio = data_cfg.get("train_validation_ratio", 0.9)

        # 1. Create DataLoaders
        print(f"[1/4] Preparing normal-only training and validation loaders...")
        manifest_dir = self.output_dir / "manifests"
        train_loader, val_loader, test_loader, meta = create_autovi_dataloaders(
            workspace_root=self.workspace_root,
            category=self.category,
            target_size=target_size,
            batch_size=batch_size,
            train_val_ratio=train_val_ratio,
            seed=seed,
            num_workers=num_workers,
            manifest_dir=manifest_dir,
        )
        print(f"      Train samples (normal): {meta['num_train']}")
        print(f"      Val samples (normal):   {meta['num_val']}")
        print(f"      Test samples (total):   {meta['num_test']}")

        # 2. Initialize Model
        print(f"[2/4] Initializing PatchCore with backbone {model_cfg.get('backbone', 'wide_resnet50_2')}...")
        device_pref = train_cfg.get("device", "auto")
        device = None if device_pref == "auto" else device_pref

        model = PatchCoreModel(
            backbone_name=model_cfg.get("backbone", "wide_resnet50_2"),
            layers=tuple(model_cfg.get("layers", ["layer2", "layer3"])),
            target_size=target_size,
            coreset_ratio=model_cfg.get("coreset_sampling_ratio", 0.1),
            num_neighbors=model_cfg.get("num_neighbors", 9),
            dimension_reduction=model_cfg.get("dimension_reduction", 128),
            chunk_size=model_cfg.get("chunk_size", 8192),
            candidate_pool_size=model_cfg.get("candidate_pool_size", None),
            device=device,
        )

        # 3. Fit Memory Bank
        start_time = time.time()
        print(f"[3/4] Extracting patch features & building coreset memory bank...")
        model.fit_memory_bank(train_loader, verbose=True)
        fit_duration = time.time() - start_time
        print(f"      Memory bank built in {fit_duration:.2f} seconds.")

        # 4. Calibrate Threshold on Normal Validation Split
        target_fpr = train_cfg.get("target_fpr", 0.05)
        print(f"[4/4] Calibrating anomaly threshold on normal validation split (Target FPR: {target_fpr})...")
        calib_threshold = model.calibrate_threshold(val_loader, target_fpr=target_fpr)
        print(f"      Calibrated Decision Threshold: {calib_threshold:.4f}")

        # Save Model Artifact
        model_path = self.model_dir / "patchcore_model.pkl"
        print(f"Saving model artifact to: {model_path}...")
        model.save(model_path)

        # Save config snapshot
        with open(self.output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        # Verification: Reload model
        print("Verifying saved model artifact by reloading in fresh instance...")
        reloaded_model = PatchCoreModel.load(model_path, device=device)
        assert reloaded_model.memory_bank is not None
        assert reloaded_model.calibrated
        print("Reload verification successful!")

        return model
