"""Unit tests for PatchCore model architecture, coreset subsampling, persistence, and event logging."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.events import DefectEventLogger
from src.models.patchcore_model import PatchCoreModel


class TestPatchCore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_greedy_kcenter_subsampling(self):
        # Create synthetic feature matrix: 100 points in 64 dimensions
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 64)).astype(np.float32)

        model = PatchCoreModel(backbone_name="wide_resnet50_2", coreset_ratio=0.1)
        selected_idx = model._greedy_kcenter_subsampling(features, target_size=10, verbose=False)
        coreset = features[selected_idx]

        self.assertEqual(len(selected_idx), 10)
        self.assertEqual(coreset.shape, (10, 64))
        # Ensure all selected vectors are distinct members of the original set
        self.assertEqual(len(np.unique(coreset, axis=0)), 10)

    def test_model_save_and_reload(self):
        model = PatchCoreModel(
            backbone_name="wide_resnet50_2",
            target_size=(256, 256),
            coreset_ratio=0.1,
            num_neighbors=5,
        )
        # Set dummy memory bank and threshold
        dummy_bank = np.random.randn(20, 1536).astype(np.float32)
        model.memory_bank = dummy_bank
        model.threshold = 4.25
        model.calibrated = True

        save_file = self.root / "model.pkl"
        model.save(save_file)
        self.assertTrue(save_file.exists())

        # Reload in a new instance
        reloaded = PatchCoreModel.load(save_file)
        self.assertEqual(reloaded.backbone_name, "wide_resnet50_2")
        self.assertEqual(reloaded.threshold, 4.25)
        self.assertTrue(reloaded.calibrated)
        self.assertEqual(reloaded.memory_bank.shape, (20, 1536))
        self.assertIsNotNone(reloaded.nn_searcher)

    def test_unknown_defect_event_serialization(self):
        event_logger = DefectEventLogger(log_dir=self.root / "events")
        event = event_logger.generate_event(
            category="pipe_clip",
            anomaly_score=15.42,
            threshold=12.0,
            is_anomaly=True,
            heatmap_path="test_heatmap.png",
            mask_path="test_mask.png",
        )

        self.assertTrue(event["event_id"].startswith("ANOMALY_"))
        self.assertEqual(event["defect_class"], "unknown")
        self.assertEqual(event["product_category"], "pipe_clip")
        self.assertEqual(event["prediction"], "anomaly")

        # Verify JSON file written to disk
        json_files = list((self.root / "events").glob("*.json"))
        self.assertEqual(len(json_files), 1)


if __name__ == "__main__":
    unittest.main()
