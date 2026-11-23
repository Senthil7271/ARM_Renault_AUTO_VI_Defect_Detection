"""Unit and regression tests for AutoVI Phase 2 detection reliability, scoring methods, and calibration."""

from pathlib import Path
import unittest
import numpy as np
from sklearn.neighbors import NearestNeighbors

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_detection_optimization import (
    compute_multi_scoring_from_patches,
    calibrate_normal_thresholds,
    compute_binary_metrics_at_threshold,
    compute_oracle_operating_points,
)


class TestPhase2ScoringAndCalibration(unittest.TestCase):
    def setUp(self):
        # Create a synthetic memory bank of 20 points in 16-D
        rng = np.random.default_rng(100)
        self.memory_bank = rng.standard_normal((20, 16)).astype(np.float32)
        self.nn_searcher = NearestNeighbors(n_neighbors=9, algorithm="brute", metric="euclidean")
        self.nn_searcher.fit(self.memory_bank)

    def test_compute_multi_scoring_from_patches(self):
        # Create 15 query patches in 16-D
        rng = np.random.default_rng(200)
        query_patches = rng.standard_normal((15, 16)).astype(np.float32)

        scores = compute_multi_scoring_from_patches(query_patches, self.nn_searcher, b_neighbors=9)

        # Check all keys are present
        expected_keys = ["raw_max", "reweighted_b9", "top3_mean", "top5_mean", "top9_mean", "logsumexp_tau1"]
        for k in expected_keys:
            self.assertIn(k, scores)
            self.assertIsInstance(scores[k], float)
            self.assertGreater(scores[k], 0.0)

        # Mathematical inequality invariants:
        # Top-3 mean >= Top-5 mean >= Top-9 mean
        self.assertGreaterEqual(scores["top3_mean"], scores["top5_mean"] - 1e-6)
        self.assertGreaterEqual(scores["top5_mean"], scores["top9_mean"] - 1e-6)
        # Raw max >= Top-3 mean
        self.assertGreaterEqual(scores["raw_max"], scores["top3_mean"] - 1e-6)

    def test_single_patch_noise_suppression(self):
        """Verify that single-patch outliers (specular reflections) are effectively suppressed by Top-K."""
        # Query where all patches are identical to bank[0] (distance 0), except 1 outlier patch far away
        query_patches = np.repeat(self.memory_bank[0:1], 16, axis=0)
        # Add high distance outlier to 1 patch
        query_patches[0] = self.memory_bank[0] + 10.0

        scores = compute_multi_scoring_from_patches(query_patches, self.nn_searcher, b_neighbors=9)

        # Raw max will be large because of the single patch outlier
        raw_max = scores["raw_max"]
        top3_mean = scores["top3_mean"]
        top9_mean = scores["top9_mean"]

        self.assertGreater(raw_max, 5.0)
        # Top-3 mean should be ~ 1/3 of raw_max because other 2 are ~0
        self.assertAlmostEqual(top3_mean, raw_max / 3.0, delta=0.5)
        # Top-9 mean should be ~ 1/9 of raw_max because other 8 are ~0
        self.assertAlmostEqual(top9_mean, raw_max / 9.0, delta=0.5)
        # Top-9 provides over 80% noise reduction relative to raw max
        self.assertLess(top9_mean, raw_max * 0.25)

    def test_calibrate_normal_thresholds_and_uncertainty(self):
        # 30 validation scores normally distributed
        rng = np.random.default_rng(300)
        val_scores = rng.normal(loc=15.0, scale=1.0, size=30)

        calib = calibrate_normal_thresholds(val_scores)

        self.assertEqual(calib["n_samples"], 30)
        self.assertAlmostEqual(calib["mean"], 15.0, delta=0.5)
        self.assertIn("empirical_q95", calib)
        self.assertIn("bootstrap_ci95_threshold", calib)
        self.assertIn("bootstrap_std", calib)
        self.assertIn("parametric_gaussian_q95", calib)

        # CI bounds validity
        ci_low, ci_high = calib["bootstrap_ci95_threshold"]
        self.assertLess(ci_low, ci_high)
        self.assertLessEqual(ci_low, calib["empirical_q95"] + 1e-5)
        self.assertGreaterEqual(ci_high, calib["empirical_q95"] - 1e-5)
        self.assertGreater(calib["bootstrap_std"], 0.0)

    def test_compute_binary_metrics_at_threshold(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([1.0, 2.0, 3.0, 4.0, 3.5, 5.0, 6.0, 7.0])
        # At threshold 3.2:
        # Predictions: [0, 0, 0, 1, 1, 1, 1, 1]
        # TP = 4, FN = 0, TN = 3, FP = 1
        metrics = compute_binary_metrics_at_threshold(y_true, scores, threshold=3.2)

        self.assertEqual(metrics["TP"], 4)
        self.assertEqual(metrics["FN"], 0)
        self.assertEqual(metrics["TN"], 3)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["Recall"], 1.0)
        self.assertEqual(metrics["Specificity"], 0.75)
        self.assertEqual(metrics["False_Alarm_Rate"], 0.25)
        self.assertAlmostEqual(metrics["Precision"], 4 / 5)
        self.assertAlmostEqual(metrics["F1"], 2 * (0.8 * 1.0) / (0.8 + 1.0))

    def test_oracle_operating_bounds(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([1.0, 2.0, 3.0, 4.0, 3.5, 5.0, 6.0, 7.0])

        oracle = compute_oracle_operating_points(y_true, scores)
        self.assertIn("oracle_optimal_f1", oracle)
        self.assertIn("oracle_youdens_j", oracle)

        opt_f1 = oracle["oracle_optimal_f1"]
        self.assertGreaterEqual(opt_f1["F1"], 0.8)
        youden = oracle["oracle_youdens_j"]
        self.assertGreaterEqual(youden["Recall"] - youden["False_Alarm_Rate"], 0.5)

    def test_phase2_report_structure_if_generated(self):
        """Verifies report structure when the phase 2 execution completes."""
        import json
        json_path = PROJECT_ROOT / "reports" / "phase2_detection_optimization.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            categories = ["pipe_clip", "pipe_staple", "engine_wiring", "tank_screw", "underbody_pipes", "underbody_screw"]
            for c in categories:
                self.assertIn(c, data)
                self.assertIn("aurocs_by_method", data[c])
                self.assertIn("raw_max", data[c]["aurocs_by_method"])
                self.assertIn("top9_mean", data[c]["aurocs_by_method"])
                self.assertIn("validation_calibration", data[c])
                self.assertIn("metrics_at_calibrated_threshold", data[c])


if __name__ == "__main__":
    unittest.main()
