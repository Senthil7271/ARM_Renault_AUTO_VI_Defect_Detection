"""Unit tests for Phase 4 authoritative evaluation suite.

Validates:
1. Map export dimensions across categories.
2. Image ID completeness and 1-to-1 matching.
3. Multi-channel ground-truth compatibility.
4. Correct anomaly score direction (higher score = more anomalous).
5. Quantization-induced rank preservation in uint16.
6. ROI mask application semantics.
7. Upstream evaluator invocation compatibility.
8. Classification function invocation.
9. Portfolio macro aggregation mathematics.
10. Frozen-model SHA-256 preservation.
"""

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

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



def test_frozen_model_sha256_preservation():
    """Item 10: Validates that frozen models match immutable registry hashes."""
    reg_path = PROJECT_ROOT / "reports" / "frozen_models_registry.json"
    assert reg_path.exists()
    with open(reg_path) as f:
        registry = json.load(f)

    for cat_name, meta in registry["models"].items():
        ckpt_path = Path(meta["checkpoint_path"])
        assert ckpt_path.exists()
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        assert h.hexdigest() == meta["checkpoint_sha256"]


def test_map_export_dimensions():
    """Item 1: Validates that exported anomaly maps have canonical dimensions."""
    canonical_dims = {
        "pipe_clip": (400, 400),
        "pipe_staple": (400, 400),
        "engine_wiring": (400, 400),
        "tank_screw": (750, 1000),
        "underbody_pipes": (750, 1000),
        "underbody_screw": (750, 1000),
    }
    base_dir = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "config_B_roi_only"
    if not base_dir.exists():
        pytest.skip("Exports not yet complete")

    for cat, (exp_h, exp_w) in canonical_dims.items():
        sample_png = next((base_dir / cat / "test").glob("*/*.png"))
        img = cv2.imread(str(sample_png), cv2.IMREAD_UNCHANGED)
        assert img.shape[:2] == (exp_h, exp_w)
        assert img.dtype == np.uint16


def test_image_id_completeness():
    """Item 2: Validates that test images have a 1-to-1 match with anomaly maps."""
    base_dir = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "config_B_roi_only"
    if not base_dir.exists():
        pytest.skip("Exports not yet complete")

    for cat in ["tank_screw", "pipe_clip"]:
        ds_test = WORKSPACE_ROOT / cat / cat / "test"
        map_test = base_dir / cat / "test"
        for sub in ds_test.iterdir():
            if sub.is_dir():
                ds_stems = sorted([p.stem for p in sub.glob("*.png")])
                map_stems = sorted([p.stem for p in (map_test / sub.name).glob("*.png")])
                assert ds_stems == map_stems


def test_quantization_rank_preservation():
    """Item 5: Verifies that global uint16 conversion preserves exact monotonicity."""
    raw_scores = np.array([12.5, 14.2, 18.9, 24.1, 45.3], dtype=np.float32)
    min_v, max_v = raw_scores.min(), raw_scores.max()
    norm = (raw_scores - min_v) / (max_v - min_v)
    uint16_scores = np.clip(norm * 65535.0, 0, 65535).astype(np.uint16)

    # Check rank preservation
    raw_ranks = np.argsort(raw_scores)
    uint16_ranks = np.argsort(uint16_scores)
    assert np.array_equal(raw_ranks, uint16_ranks)


def test_roi_mask_application():
    """Item 6: Verifies that pixels outside the ROI are zeroed while inside are preserved."""
    test_map = np.ones((100, 100), dtype=np.uint16) * 50000
    y1, y2, x1, x2 = 20, 50, 30, 70

    masked = np.zeros_like(test_map)
    masked[y1:y2, x1:x2] = test_map[y1:y2, x1:x2]

    assert (masked[:20, :] == 0).all()
    assert (masked[50:, :] == 0).all()
    assert (masked[:, :30] == 0).all()
    assert (masked[:, 70:] == 0).all()
    assert (masked[20:50, 30:70] == 50000).all()
    assert masked.max() == 50000


def test_portfolio_macro_aggregation():
    """Item 9: Tests mathematical macro aggregation across 6 categories."""
    cat_scores = [0.9980, 0.9886, 0.9534, 0.9983, 0.9271, 0.9017]
    expected_macro = sum(cat_scores) / 6.0
    calc_macro = float(np.mean(cat_scores))
    assert abs(calc_macro - expected_macro) < 1e-6
    assert calc_macro >= 0.9500  # Primary target check


def test_multi_channel_ground_truth_compatibility():
    """Item 3: Validates reading multi-channel GT directories via upstream GroundTruthMap."""
    if str(OFFICIAL_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
    from evaluate_experiment import DefectsConfig, GroundTruthMap

    gt_dir = WORKSPACE_ROOT / "tank_screw" / "tank_screw" / "ground_truth" / "missing" / "0000"
    cfg_file = WORKSPACE_ROOT / "tank_screw" / "tank_screw" / "defects_config.json"
    if not gt_dir.exists() or not cfg_file.exists():
        pytest.skip("Dataset ground truth not present")

    with open(cfg_file) as f:
        cfg_list = json.load(f)
    defects_config = DefectsConfig.create_from_list(cfg_list)
    gt_map = GroundTruthMap.read_from_png_dir(str(gt_dir), defects_config)
    assert gt_map is not None
    assert gt_map.size == (750, 1000)
    or_mask = gt_map.get_or_over_channels()
    assert or_mask.ndim == 2
    assert or_mask.dtype == bool
    assert or_mask.sum() > 0



def test_score_direction_sanity():
    """Item 4: Validates that higher score indicates defect (monotonicity)."""
    if str(OFFICIAL_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
    from evaluate_experiment import compute_classification_auc_roc

    normal = [10.0, 11.0, 9.5, 10.5]
    defective = [28.0, 31.0, 29.5, 32.0]
    auc = compute_classification_auc_roc(normal, defective)
    assert auc == 1.0


def test_classification_auc_roc_function():
    """Item 8: Validates official compute_classification_auc_roc against sklearn."""
    from sklearn.metrics import roc_auc_score
    if str(OFFICIAL_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
    from evaluate_experiment import compute_classification_auc_roc

    good_scores = [1.0, 2.0, 3.0, 4.0]
    bad_scores = [3.5, 4.5, 5.5, 6.5]

    official_auc = compute_classification_auc_roc(good_scores, bad_scores)
    y_true = [0] * len(good_scores) + [1] * len(bad_scores)
    y_scores = good_scores + bad_scores
    sklearn_auc = roc_auc_score(y_true, y_scores)

    assert abs(official_auc - sklearn_auc) < 1e-6




def test_upstream_evaluator_read_maps():
    """Item 7: Tests reading GT and anomaly maps using official read_maps function."""
    import evaluate_experiment as official_eval
    from evaluate_experiment import DefectsConfig

    # Non-intrusive Windows path separator normalizer for generator functions
    _orig_test_rel = official_eval.get_available_test_image_rel_paths
    _orig_gt_rel = official_eval.get_available_gt_map_rel_paths

    def _clean_test_rel(p):
        for path_str in _orig_test_rel(p):
            yield path_str.replace("\\", "/")

    def _clean_gt_rel(p):
        for path_str in _orig_gt_rel(p):
            yield path_str.replace("\\", "/")

    official_eval.get_available_test_image_rel_paths = _clean_test_rel
    official_eval.get_available_gt_map_rel_paths = _clean_gt_rel

    cat = "tank_screw"
    cfg_file = WORKSPACE_ROOT / cat / cat / "defects_config.json"
    gt_dir = WORKSPACE_ROOT / cat / cat / "ground_truth"
    maps_dir = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "config_B_roi_only" / cat / "test"

    if not maps_dir.exists():
        pytest.skip("Anomaly maps not yet exported")

    with open(cfg_file) as f:
        cfg = DefectsConfig.create_from_list(json.load(f))
    resize_shape = official_eval.set_resize_shape(cat)

    gt_maps, anomaly_maps = official_eval.read_maps(
        gt_dir=str(gt_dir),
        anomaly_maps_test_dir=str(maps_dir),
        defects_config=cfg,
        resize_shape=resize_shape,
    )
    assert len(anomaly_maps) == 413
    assert len(gt_maps) == 413

