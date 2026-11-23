"""Unit tests for ground-truth mask handling, multi-mask union, and resizing."""

from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.masks import MaskProcessor


class TestMasks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create synthetic category structure
        self.cat_root = self.root / "category"
        self.gt_dir = self.cat_root / "ground_truth" / "multiple" / "0001"
        self.gt_dir.mkdir(parents=True, exist_ok=True)

        # Create two distinct masks for the same image (e.g. left defect & right defect)
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[10:30, 10:30] = 255  # top-left region

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[60:80, 60:80] = 254  # bottom-right region with multiclass value 254

        cv2.imwrite(str(self.gt_dir / "0000.png"), mask1)
        cv2.imwrite(str(self.gt_dir / "0001.png"), mask2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_mask_paths(self):
        proc = MaskProcessor()
        paths = proc.find_mask_paths(self.cat_root, "multiple", "0001")
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p.exists() for p in paths))

    def test_multi_mask_union(self):
        proc = MaskProcessor(target_size=(100, 100))
        paths = proc.find_mask_paths(self.cat_root, "multiple", "0001")

        bin_mask, multi_mask = proc.load_combined_mask(paths)

        # In binary mask, both regions should be 1
        self.assertEqual(bin_mask[20, 20], 1)
        self.assertEqual(bin_mask[70, 70], 1)
        self.assertEqual(bin_mask[0, 0], 0)

        # In multiclass mask, values should be preserved
        self.assertEqual(multi_mask[20, 20], 255)
        self.assertEqual(multi_mask[70, 70], 254)

    def test_nearest_neighbor_resizing(self):
        # When resizing, binary mask must strictly contain only {0, 1}
        proc = MaskProcessor(target_size=(256, 256))
        paths = proc.find_mask_paths(self.cat_root, "multiple", "0001")
        bin_mask, _ = proc.load_combined_mask(paths)

        self.assertEqual(bin_mask.shape, (256, 256))
        unique_vals = set(np.unique(bin_mask))
        self.assertTrue(unique_vals.issubset({0, 1}), f"Resized mask has non-binary values: {unique_vals}")

    def test_empty_mask_for_normal_sample(self):
        proc = MaskProcessor(target_size=(128, 128))
        bin_mask, multi_mask = proc.load_combined_mask([], original_shape=(400, 400))
        self.assertEqual(bin_mask.shape, (128, 128))
        self.assertEqual(np.count_nonzero(bin_mask), 0)


if __name__ == "__main__":
    unittest.main()
