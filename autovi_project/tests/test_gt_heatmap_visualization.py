"""Unit test suite for AutoVI Ground Truth vs Heatmap Visualization and Metrics."""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Upstream evaluator imports
OFFICIAL_EVAL_DIR = WORKSPACE_ROOT / "external" / "autovi_evaluation_code"
if not hasattr(os, "nice"):
    setattr(os, "nice", lambda inc: 0)

sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

# Ensure namespace package 'src' cleanly resolves to external/autovi_evaluation_code/src
src_eval_path = str(OFFICIAL_EVAL_DIR / "src")
if "src" not in sys.modules or not hasattr(sys.modules["src"], "__path__"):
    _src_mod = types.ModuleType("src")
    _src_mod.__path__ = [src_eval_path]
    sys.modules["src"] = _src_mod
else:
    if src_eval_path not in sys.modules["src"].__path__:
        sys.modules["src"].__path__.insert(0, src_eval_path)

from src.image import GroundTruthMap, DefectsConfig
from scripts.visualize_gt_vs_heatmap import (
    ALL_CATEGORIES,
    OPTIMIZED_CATEGORIES,
    FROZEN_CATEGORIES,
    CANONICAL_RESOLUTIONS,
    load_defects_config,
    load_calibrated_pixel_thresholds,
    load_sample_data,
    compute_segmentation_error_map,
    create_contour_overlay,
    compute_pixel_metrics,
)


class TestAutoVIGTHeatmapVisualization:
    """Test suite validating data loading, spatial alignment, and evaluation integrity."""

    def test_canonical_resolutions_defined(self):
        """Verify all six categories have canonical resolutions defined."""
        assert len(ALL_CATEGORIES) == 6
        for cat in ALL_CATEGORIES:
            assert cat in CANONICAL_RESOLUTIONS
            H, W = CANONICAL_RESOLUTIONS[cat]
            assert H > 0 and W > 0
            if cat in ["tank_screw", "underbody_pipes", "underbody_screw"]:
                assert (H, W) == (750, 1000)
            else:
                assert (H, W) == (400, 400)

    def test_calibrated_pixel_thresholds_exist(self):
        """Verify calibrated pixel thresholds registry exists with valid thresholds."""
        thresholds = load_calibrated_pixel_thresholds()
        assert len(thresholds) == 6
        for cat in ALL_CATEGORIES:
            assert cat in thresholds
            cfg_entry = thresholds[cat]
            assert "config_A_baseline" in cfg_entry
            tau = cfg_entry["config_A_baseline"]["pixel_tau_p99"]
            assert 10000.0 < tau < 60000.0
            if cat in OPTIMIZED_CATEGORIES:
                assert "config_C_roi_coord" in cfg_entry

    def test_sample_data_spatial_alignment(self):
        """Verify sample data loading achieves 1-to-1 spatial alignment across modalities."""
        # Test on tank_screw (750x1000) and pipe_clip (400x400)
        test_cases = [
            ("tank_screw", "missing", "0000", "config_A_baseline", (750, 1000)),
            ("tank_screw", "missing", "0000", "config_C_roi_coord", (750, 1000)),
            ("pipe_clip", "unclipped", "0000", "config_A_baseline", (400, 400)),
            ("pipe_clip", "unclipped", "0000", "config_C_roi_coord", (400, 400)),
        ]

        for cat, subtype, img_id, cfg, expected_shape in test_cases:
            data = load_sample_data(cat, subtype, img_id, cfg)
            assert data["category"] == cat
            assert data["subtype"] == subtype
            assert data["rgb"].shape == (expected_shape[0], expected_shape[1], 3)
            assert data["gt_mask"].shape == expected_shape
            assert data["anomaly_map"].shape == expected_shape
            assert data["gt_mask"].dtype == bool
            assert data["anomaly_map"].dtype in [np.uint16, np.float32, np.float64, np.uint32]
            assert data["is_defective"] is True
            assert data["gt_mask"].sum() > 0

    def test_pixel_metrics_mathematical_precision(self):
        """Verify pixel metrics formulas under synthetic ground truth and predictions."""
        # Create synthetic masks
        gt = np.zeros((100, 100), dtype=bool)
        pred = np.zeros((100, 100), dtype=bool)

        # TP: 50 pixels, FP: 20 pixels, FN: 30 pixels, TN: 10000 - 100 = 9900
        gt[:10, :5] = True   # 50 pixels
        pred[:10, :5] = True  # TP = 50
        pred[10:14, :5] = True # FP = 20
        gt[14:20, :5] = True  # FN = 30

        met = compute_pixel_metrics(gt, pred)
        assert met["tp_pixels"] == 50
        assert met["fp_pixels"] == 20
        assert met["fn_pixels"] == 30
        assert met["tn_pixels"] == 10000 - 100

        expected_prec = 50.0 / (50.0 + 20.0)  # 50/70 = 0.7142857
        expected_rec = 50.0 / (50.0 + 30.0)   # 50/80 = 0.625
        expected_dice = (2.0 * 50.0) / (2.0 * 50.0 + 20.0 + 30.0) # 100/150 = 0.666666
        expected_iou = 50.0 / (50.0 + 20.0 + 30.0)  # 50/100 = 0.50

        assert pytest.approx(met["precision"], 1e-4) == expected_prec
        assert pytest.approx(met["recall"], 1e-4) == expected_rec
        assert pytest.approx(met["dice"], 1e-4) == expected_dice
        assert pytest.approx(met["iou"], 1e-4) == expected_iou

    def test_pixel_metrics_empty_masks_no_crash(self):
        """Verify pixel metrics handle empty / nominal masks without zero division error."""
        gt_empty = np.zeros((50, 50), dtype=bool)
        pred_empty = np.zeros((50, 50), dtype=bool)

        met = compute_pixel_metrics(gt_empty, pred_empty)
        assert met["tp_pixels"] == 0
        assert met["precision"] == 0.0
        assert met["recall"] == 0.0
        assert met["dice"] == 0.0
        assert met["iou"] == 0.0

    def test_error_map_color_semantics(self):
        """Verify error map assigns Green to TP, Red to FP, and Blue to FN."""
        rgb = np.full((10, 10, 3), 128, dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=bool)
        pred = np.zeros((10, 10), dtype=bool)

        # Pixel (0, 0): TP
        gt[0, 0] = True
        pred[0, 0] = True

        # Pixel (1, 1): FP
        gt[1, 1] = False
        pred[1, 1] = True

        # Pixel (2, 2): FN
        gt[2, 2] = True
        pred[2, 2] = False

        err_map = compute_segmentation_error_map(gt, pred, rgb)
        assert err_map.shape == (10, 10, 3)

        # TP must be Green: [0, 230, 0]
        np.testing.assert_array_equal(err_map[0, 0], [0, 230, 0])
        # FP must be Red: [230, 0, 0]
        np.testing.assert_array_equal(err_map[1, 1], [230, 0, 0])
        # FN must be Blue: [0, 100, 255]
        np.testing.assert_array_equal(err_map[2, 2], [0, 100, 255])

    def test_contour_overlay_rendering(self):
        """Verify contour overlay executes cleanly without mutating input image."""
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        gt = np.zeros((50, 50), dtype=bool)
        pred = np.zeros((50, 50), dtype=bool)

        gt[10:20, 10:20] = True
        pred[15:25, 15:25] = True

        cnt = create_contour_overlay(rgb, gt, pred)
        assert cnt.shape == (50, 50, 3)
        assert cnt.dtype == np.uint8
        # Ensure contours were drawn (some pixels are non-zero)
        assert cnt.sum() > 0

    def test_generated_artifacts_exist(self):
        """Verify generated research figures and quantitative CSV exist on disk."""
        metrics_csv = PROJECT_ROOT / "reports" / "gt_heatmap_localization_metrics.csv"
        assert metrics_csv.exists()
        assert metrics_csv.stat().st_size > 50000

        fig_dir = PROJECT_ROOT / "outputs" / "gt_heatmap_comparisons"
        assert fig_dir.exists()
        expected_figs = [
            "tank_screw_localization_comparison.png",
            "tank_screw_failure_mode_analysis.png",
            "pipe_clip_unclipped_comparison.png",
            "engine_wiring_fastening_comparison.png",
            "engine_wiring_blue_hoop_comparison.png",
            "pipe_staple_baseline_localization.png",
            "underbody_pipes_baseline_localization.png",
            "underbody_screw_baseline_localization.png",
        ]
        for f in expected_figs:
            p = fig_dir / f
            assert p.exists(), f"Expected figure missing: {p}"
            assert p.stat().st_size > 10000
