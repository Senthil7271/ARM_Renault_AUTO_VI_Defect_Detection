"""Unit and integration tests for ROIPatchCoreModel (Phase 4 Candidate B)."""

import shutil
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.roi_patchcore import ROIPatchCoreModel


class DummyImageDataset(Dataset):
    def __init__(self, num_samples: int = 10, img_size: tuple = (256, 256)):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate pseudo-nominal image with central bright block
        h, w = self.img_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        y1, y2 = int(0.3 * h), int(0.7 * h)
        x1, x2 = int(0.3 * w), int(0.7 * w)
        img[y1:y2, x1:x2] = np.random.randint(180, 240, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return {"image": t_img, "path": f"dummy_{idx}.png"}


class TestROIPatchCore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.device = "cpu"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_roi_derivation_from_normal(self):
        """Test normal-only ROI derivation using synthetic sample files."""
        sample_paths = []
        for i in range(5):
            p = self.temp_dir / f"norm_{i}.png"
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            img[90:170, 100:180] = 200
            Image.fromarray(img).save(p)
            sample_paths.append(p)

        roi = ROIPatchCoreModel.derive_normal_roi_from_paths(sample_paths, target_size=(256, 256))
        ymin, xmin, ymax, xmax = roi
        self.assertTrue(0.0 <= ymin < ymax <= 1.0)
        self.assertTrue(0.0 <= xmin < xmax <= 1.0)
        # Component is roughly in [0.35, 0.39, 0.66, 0.70]
        self.assertLess(ymin, 0.45)
        self.assertGreater(ymax, 0.60)

    def test_feature_extraction_with_coordinates(self):
        """Test multi-scale feature extraction with 2D coordinate conditioning."""
        model = ROIPatchCoreModel(
            backbone_name="resnet18", # lightweight backbone for CPU test
            layers=("layer2", "layer3"),
            target_size=(128, 128),
            device=self.device,
            roi_box=(0.25, 0.25, 0.75, 0.75),
            coord_scale=0.05,
        )
        dummy_x = torch.randn(2, 3, 128, 128)
        patches, grid_shape = model.extract_patch_features(dummy_x, roi_only=True)

        self.assertIsInstance(patches, torch.Tensor)
        # Layer2 (128) + Layer3 (256) = 384 + 2 (coords) = 386 channels
        self.assertEqual(patches.shape[-1], 384 + 2)
        self.assertGreater(len(patches), 0)

    def test_full_pipeline_fit_calibrate_predict_serialize(self):
        """Test complete fitting, normal-only calibration, inference, and serialization."""
        train_ds = DummyImageDataset(num_samples=8, img_size=(128, 128))
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)

        val_ds = DummyImageDataset(num_samples=4, img_size=(128, 128))
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

        model = ROIPatchCoreModel(
            backbone_name="resnet18",
            layers=("layer2", "layer3"),
            target_size=(128, 128),
            coreset_ratio=0.2,
            device=self.device,
            roi_box=(0.3, 0.3, 0.7, 0.7),
            coord_scale=0.05,
        )

        # 1. Fit memory bank
        model.fit_memory_bank(train_loader, verbose=False)
        self.assertIsNotNone(model.memory_bank)
        self.assertGreater(len(model.memory_bank), 0)

        # 2. Calibrate threshold on normal validation only
        threshold = model.calibrate_threshold(val_loader, target_fpr=0.05)
        self.assertTrue(model.calibrated)
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0.0)

        # 3. Predict anomaly on a single image
        test_img = torch.randn(1, 3, 128, 128)
        score, amap = model.predict_anomaly(test_img)
        self.assertIsInstance(score, float)
        self.assertEqual(amap.shape, (128, 128))
        self.assertTrue(np.all(np.isfinite(amap)))

        # 4. Save and reload checkpoint
        ckpt_path = self.temp_dir / "roi_patchcore.pt"
        model.save_roi_checkpoint(ckpt_path, metadata={"test": "ok"})
        self.assertTrue(ckpt_path.exists())

        reloaded_model = ROIPatchCoreModel.load_roi_checkpoint(ckpt_path, device=self.device)
        self.assertTrue(reloaded_model.calibrated)
        self.assertEqual(reloaded_model.threshold, threshold)
        self.assertEqual(reloaded_model.roi_box, (0.3, 0.3, 0.7, 0.7))

        # 5. Check prediction consistency between original and reloaded models
        reloaded_score, reloaded_amap = reloaded_model.predict_anomaly(test_img)
        self.assertAlmostEqual(score, reloaded_score, places=4)
        np.testing.assert_allclose(amap, reloaded_amap, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
