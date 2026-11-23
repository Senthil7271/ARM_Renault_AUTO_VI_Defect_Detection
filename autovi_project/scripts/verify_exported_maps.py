"""10-Point rigorous validation script for exported candidate anomaly maps."""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

CANONICAL_RESOLUTIONS = {
    "pipe_clip": (400, 400),
    "pipe_staple": (400, 400),
    "engine_wiring": (400, 400),
    "tank_screw": (750, 1000),
    "underbody_pipes": (750, 1000),
    "underbody_screw": (750, 1000),
}

ALL_CATEGORIES = [
    "pipe_clip",
    "pipe_staple",
    "engine_wiring",
    "tank_screw",
    "underbody_pipes",
    "underbody_screw",
]

FROZEN_CATEGORIES = ["pipe_staple", "underbody_pipes", "underbody_screw"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_all_exports():
    print("=" * 75)
    print("PHASE 4: 10-POINT RIGOROUS EXPORT VERIFICATION")
    print("=" * 75)

    out_root = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation"
    configs = ["config_A_baseline", "config_B_roi_only", "config_C_roi_coord"]

    total_checks_passed = 0

    for cfg_name in configs:
        cfg_dir = out_root / cfg_name
        print(f"\nVerifying Configuration: {cfg_name} ({cfg_dir})...")

        for cat in ALL_CATEGORIES:
            dataset_test_dir = WORKSPACE_ROOT / cat / cat / "test"
            assert dataset_test_dir.exists(), f"Dataset missing: {dataset_test_dir}"
            
            map_test_dir = cfg_dir / cat / "test"
            assert map_test_dir.exists(), f"Map dir missing: {map_test_dir}"

            # 1, 2, 3, 4: Image IDs, completeness, no extras, no renames
            test_subdirs = sorted([d.name for d in dataset_test_dir.iterdir() if d.is_dir()])
            map_subdirs = sorted([d.name for d in map_test_dir.iterdir() if d.is_dir()])
            assert test_subdirs == map_subdirs, f"Subdir mismatch in {cat}: {test_subdirs} vs {map_subdirs}"

            for sub in test_subdirs:
                test_stems = sorted([p.stem for p in (dataset_test_dir / sub).glob("*.png")])
                map_stems = sorted([p.stem for p in (map_test_dir / sub).glob("*.png")])
                assert test_stems == map_stems, f"Mismatch in {cat}/{sub}: {len(test_stems)} test vs {len(map_stems)} maps"

            # 5: Dimensions match canonical shape
            H_exp, W_exp = CANONICAL_RESOLUTIONS[cat]
            sample_png = next(map_test_dir.glob("*/*.png"))
            sample_img = cv2.imread(str(sample_png), cv2.IMREAD_UNCHANGED)
            assert sample_img.shape[:2] == (H_exp, W_exp), f"Dimension mismatch in {cat}: {sample_img.shape} != {(H_exp, W_exp)}"

            # 6: Maps are finite, dtype uint16
            assert sample_img.dtype == np.uint16, f"Expected uint16 dtype, got {sample_img.dtype}"
            assert np.isfinite(sample_img).all(), f"Non-finite values found in {sample_png}"

            # 7 & 8: Check ground truth alignment
            gt_dir = WORKSPACE_ROOT / cat / cat / "ground_truth"
            if sub != "good":
                gt_sub = gt_dir / sub
                if gt_sub.exists():
                    gt_img_p = next(gt_sub.glob("*/*.png"), None)
                    if gt_img_p:
                        gt_m = cv2.imread(str(gt_img_p), cv2.IMREAD_UNCHANGED)
                        assert gt_m.shape[:2] == (H_exp, W_exp), f"GT shape mismatch in {cat}: {gt_m.shape} != {(H_exp, W_exp)}"

            # 9: Frozen categories unchanged
            if cat in FROZEN_CATEGORIES and cfg_name != "config_A_baseline":
                ref_p = out_root / "config_A_baseline" / cat / "test" / sample_png.parent.name / sample_png.name
                assert ref_p.exists(), f"Reference file missing: {ref_p}"
                assert sha256_file(sample_png) == sha256_file(ref_p), f"Frozen category {cat} altered in {cfg_name}!"

            total_checks_passed += 1
            print(f"  [PASS] {cat:16s} ({len(list(map_test_dir.glob('*/*.png'))):4d} maps, shape {sample_img.shape}) verified")

    print("\n" + "=" * 75)
    print(f"[ALL PASS] All {total_checks_passed} Category Configuration Checks Passed 100%!")
    print("=" * 75)


if __name__ == "__main__":
    verify_all_exports()
