"""Unit tests for dataset detection, loading, split reproducibility, and leakage prevention."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.audit import DatasetAuditor
from src.dataset.loader import AutoVIDataset, create_autovi_dataloaders


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

        # Create synthetic category with nested structure
        self.cat_name = "test_cat"
        self.cat_dir = self.workspace / self.cat_name / self.cat_name
        self.train_good = self.cat_dir / "train" / "good"
        self.test_good = self.cat_dir / "test" / "good"
        self.test_defect = self.cat_dir / "test" / "crack"
        self.gt_defect = self.cat_dir / "ground_truth" / "crack"

        for p in [self.train_good, self.test_good, self.test_defect, self.gt_defect]:
            p.mkdir(parents=True, exist_ok=True)

        # Create 10 train images (all normal)
        for i in range(10):
            img = Image.fromarray((np.ones((64, 64, 3)) * (i * 20)).astype(np.uint8))
            img.save(self.train_good / f"{i:04d}.png")

        # Create 5 test normal images
        for i in range(5):
            img = Image.fromarray((np.ones((64, 64, 3)) * 50).astype(np.uint8))
            img.save(self.test_good / f"{i:04d}.png")

        # Create 5 test defective images + masks
        for i in range(5):
            img = Image.fromarray((np.ones((64, 64, 3)) * 200).astype(np.uint8))
            img.save(self.test_defect / f"{i:04d}.png")

            mask_dir = self.gt_defect / f"{i:04d}"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask = Image.fromarray((np.ones((64, 64)) * 255).astype(np.uint8))
            mask.save(mask_dir / "0000.png")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_category_detection_and_resolution(self):
        auditor = DatasetAuditor(self.workspace)
        resolved = auditor.resolve_category_path(self.cat_name)
        self.assertEqual(resolved, self.cat_dir)
        self.assertTrue(resolved.exists())

    def test_normal_only_training_loader(self):
        train_loader, val_loader, test_loader, meta = create_autovi_dataloaders(
            workspace_root=self.workspace,
            category=self.cat_name,
            target_size=(32, 32),
            batch_size=2,
            train_val_ratio=0.8,
            seed=42,
        )
        self.assertEqual(meta["num_train"], 8)
        self.assertEqual(meta["num_val"], 2)
        self.assertEqual(meta["num_test"], 10)

        # Verify all training samples are strictly normal (label = 0)
        for batch in train_loader:
            labels = batch["label"].numpy()
            self.assertTrue(np.all(labels == 0))
            for dt in batch["defect_type"]:
                self.assertEqual(dt, "good")

    def test_split_reproducibility(self):
        manifest_dir = Path(self.temp_dir.name) / "manifests"
        _, _, _, meta1 = create_autovi_dataloaders(
            workspace_root=self.workspace,
            category=self.cat_name,
            target_size=(32, 32),
            train_val_ratio=0.8,
            seed=42,
            manifest_dir=manifest_dir,
        )

        with open(manifest_dir / f"{self.cat_name}_split_manifest.json") as f:
            man1 = json.load(f)

        _, _, _, meta2 = create_autovi_dataloaders(
            workspace_root=self.workspace,
            category=self.cat_name,
            target_size=(32, 32),
            train_val_ratio=0.8,
            seed=42,
            manifest_dir=manifest_dir,
        )

        with open(manifest_dir / f"{self.cat_name}_split_manifest.json") as f:
            man2 = json.load(f)

        self.assertEqual(man1["train_images"], man2["train_images"])
        self.assertEqual(man1["val_images"], man2["val_images"])

    def test_train_val_no_leakage(self):
        manifest_dir = Path(self.temp_dir.name) / "manifests"
        create_autovi_dataloaders(
            workspace_root=self.workspace,
            category=self.cat_name,
            target_size=(32, 32),
            train_val_ratio=0.8,
            seed=42,
            manifest_dir=manifest_dir,
        )
        with open(manifest_dir / f"{self.cat_name}_split_manifest.json") as f:
            manifest = json.load(f)

        train_set = set(manifest["train_images"])
        val_set = set(manifest["val_images"])
        overlap = train_set.intersection(val_set)
        self.assertEqual(len(overlap), 0, "Train and validation splits must have zero overlap")


if __name__ == "__main__":
    unittest.main()
