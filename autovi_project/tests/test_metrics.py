"""Unit tests for evaluation metrics (Image AUROC, Pixel AUROC, AUPRO, IoU, Dice)."""

import unittest
import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_au_pro, compute_image_metrics, compute_pixel_metrics


class TestMetrics(unittest.TestCase):
    def test_image_metrics_perfect_separation(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9, 0.85, 0.95])
        threshold = 0.5

        metrics = compute_image_metrics(y_true, y_score, threshold)
        self.assertAlmostEqual(metrics["image_auroc"], 1.0)
        self.assertAlmostEqual(metrics["image_ap"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["fpr"], 0.0)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 4)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 4)

    def test_pixel_metrics_synthetic(self):
        # 2 images of size 10x10
        masks_true = np.zeros((2, 10, 10), dtype=int)
        masks_true[1, 2:6, 2:6] = 1  # Anomaly region in second image

        # Predicted heatmap that perfectly highlights the anomaly
        maps_pred = np.zeros((2, 10, 10), dtype=float)
        maps_pred[1, 2:6, 2:6] = 2.0  # High score inside anomaly

        metrics = compute_pixel_metrics(masks_true, maps_pred, threshold=1.0)
        self.assertAlmostEqual(metrics["pixel_auroc"], 1.0)
        self.assertAlmostEqual(metrics["pixel_iou"], 1.0)
        self.assertAlmostEqual(metrics["pixel_dice"], 1.0)

    def test_au_pro_perfect_overlap(self):
        masks_true = np.zeros((1, 20, 20), dtype=int)
        masks_true[0, 5:10, 5:10] = 1  # 5x5 connected component

        maps_pred = np.zeros((1, 20, 20), dtype=float)
        maps_pred[0, 5:10, 5:10] = 5.0  # high score

        au_pro = compute_au_pro(masks_true, maps_pred, max_fpr=0.3)
        self.assertGreater(au_pro, 0.9)


if __name__ == "__main__":
    unittest.main()
