"""Export authoritative anomaly maps for AutoVI Phase 4 three-configuration evaluation.

Generates:
1. config_A_baseline: Exact original full-frame PatchCore maps for all 6 categories.
2. config_B_roi_only: Deployable Normal-Only ROI-masked maps for active categories; frozen categories copied byte-for-byte.
3. config_C_roi_coord: Deployable Normal ROI + coordinate conditioned maps for active categories; frozen categories copied byte-for-byte.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CANONICAL_RESOLUTIONS = {
    "pipe_clip": (400, 400),
    "pipe_staple": (400, 400),
    "engine_wiring": (400, 400),
    "tank_screw": (750, 1000),       # (H, W) in OpenCV / NumPy
    "underbody_pipes": (750, 1000),
    "underbody_screw": (750, 1000),
}

FROZEN_CATEGORIES = ["pipe_staple", "underbody_pipes", "underbody_screw"]
ACTIVE_CATEGORIES = ["tank_screw", "pipe_clip", "engine_wiring"]


def export_all_configurations():
    print("=" * 75)
    print("PHASE 4: EXPORTING CANDIDATE ANOMALY MAPS FOR AUTHORITATIVE BENCHMARK")
    print("=" * 75)

    base_maps_dir = PROJECT_ROOT / "outputs" / "official_autovi" / "anomaly_maps"
    out_root = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation"

    config_A_dir = out_root / "config_A_baseline"
    config_B_dir = out_root / "config_B_roi_only"
    config_C_dir = out_root / "config_C_roi_coord"

    for d in [config_A_dir, config_B_dir, config_C_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load versioned deployable ROI configurations
    roi_config_path = PROJECT_ROOT / "configs" / "roi_configurations.json"
    with open(roi_config_path, "r", encoding="utf-8") as f:
        roi_meta = json.load(f)

    # -------------------------------------------------------------
    # 1. Configuration A (Baseline): Copy exact original maps
    # -------------------------------------------------------------
    print("\n[CONFIG A] Assembling Baseline PatchCore anomaly maps...")
    all_cats = ACTIVE_CATEGORIES + FROZEN_CATEGORIES
    for cat in all_cats:
        src_cat_test = base_maps_dir / cat / "test"
        dst_cat_test = config_A_dir / cat / "test"
        dst_cat_test.mkdir(parents=True, exist_ok=True)

        for sub in src_cat_test.iterdir():
            if not sub.is_dir():
                continue
            dst_sub = dst_cat_test / sub.name
            dst_sub.mkdir(parents=True, exist_ok=True)
            for png_p in sub.glob("*.png"):
                dst_png = dst_sub / png_p.name
                if not dst_png.exists():
                    shutil.copy2(png_p, dst_png)
        count_a = len(list(dst_cat_test.glob("*/*.png")))
        print(f"  - {cat:16s}: {count_a} maps copied to Config A")

    # -------------------------------------------------------------
    # 2. Configuration B (ROI-Only): Mask active categories to normal ROI
    # -------------------------------------------------------------
    print("\n[CONFIG B] Generating Deployable Normal ROI-only anomaly maps...")
    for cat in all_cats:
        src_cat_test = config_A_dir / cat / "test"
        dst_cat_test = config_B_dir / cat / "test"
        dst_cat_test.mkdir(parents=True, exist_ok=True)

        if cat in FROZEN_CATEGORIES:
            # Frozen categories: copy byte-for-byte unchanged
            for sub in src_cat_test.iterdir():
                if not sub.is_dir():
                    continue
                dst_sub = dst_cat_test / sub.name
                dst_sub.mkdir(parents=True, exist_ok=True)
                for png_p in sub.glob("*.png"):
                    shutil.copy2(png_p, dst_sub / png_p.name)
            count_b = len(list(dst_cat_test.glob("*/*.png")))
            print(f"  - {cat:16s} [FROZEN]: {count_b} maps preserved unchanged from Config A")
        else:
            # Active categories: apply Deployable Normal ROI mask
            box = roi_meta["categories"][cat]["deployable_normal_only_roi"]["box"]
            H_c, W_c = CANONICAL_RESOLUTIONS[cat]
            y1, y2 = int(box[0] * H_c), int(box[2] * H_c)
            x1, x2 = int(box[1] * W_c), int(box[3] * W_c)

            print(f"  - {cat:16s} [ACTIVE]: Applying Deployable ROI [{y1}:{y2}, {x1}:{x2}] (H={H_c}, W={W_c})...")
            for sub in src_cat_test.iterdir():
                if not sub.is_dir():
                    continue
                dst_sub = dst_cat_test / sub.name
                dst_sub.mkdir(parents=True, exist_ok=True)
                for png_p in sub.glob("*.png"):
                    orig_map = cv2.imread(str(png_p), cv2.IMREAD_UNCHANGED)
                    # Ensure canonical resolution
                    if orig_map.shape[:2] != (H_c, W_c):
                        orig_map = cv2.resize(orig_map, (W_c, H_c), interpolation=cv2.INTER_LINEAR)
                    
                    # Create ROI-masked map
                    # Retain values inside ROI; zero outside ROI to preserve low-score semantics
                    roi_map = np.zeros_like(orig_map)
                    roi_map[y1:y2, x1:x2] = orig_map[y1:y2, x1:x2]
                    cv2.imwrite(str(dst_sub / png_p.name), roi_map)

            count_b = len(list(dst_cat_test.glob("*/*.png")))
            print(f"    Exported {count_b} ROI-masked maps for {cat}")

    # -------------------------------------------------------------
    # 3. Configuration C (ROI + Coordinate Conditioning)
    # -------------------------------------------------------------
    print("\n[CONFIG C] Generating ROI + Coordinate Conditioning anomaly maps...")
    for cat in all_cats:
        src_cat_test = config_B_dir / cat / "test"
        dst_cat_test = config_C_dir / cat / "test"
        dst_cat_test.mkdir(parents=True, exist_ok=True)

        if cat in FROZEN_CATEGORIES:
            # Frozen categories: copy byte-for-byte unchanged
            for sub in src_cat_test.iterdir():
                if not sub.is_dir():
                    continue
                dst_sub = dst_cat_test / sub.name
                dst_sub.mkdir(parents=True, exist_ok=True)
                for png_p in sub.glob("*.png"):
                    shutil.copy2(png_p, dst_sub / png_p.name)
            count_c = len(list(dst_cat_test.glob("*/*.png")))
            print(f"  - {cat:16s} [FROZEN]: {count_c} maps preserved unchanged from Config A")
        else:
            # Active categories: apply coordinate spatial weighting to ROI maps
            box = roi_meta["categories"][cat]["deployable_normal_only_roi"]["box"]
            H_c, W_c = CANONICAL_RESOLUTIONS[cat]
            y1, y2 = int(box[0] * H_c), int(box[2] * H_c)
            x1, x2 = int(box[1] * W_c), int(box[3] * W_c)

            # Center coordinates of the component
            cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
            # 2D Gaussian spatial falloff weight centered on nominal component
            yy, xx = np.mgrid[0:H_c, 0:W_c]
            sigma_y = max(10, (y2 - y1) / 2.0)
            sigma_x = max(10, (x2 - x1) / 2.0)
            dist_sq = ((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2
            # Mild spatial regularization: within component envelope weight is near 1.0
            spatial_weight = np.exp(-0.2 * dist_sq).astype(np.float32)

            for sub in src_cat_test.iterdir():
                if not sub.is_dir():
                    continue
                dst_sub = dst_cat_test / sub.name
                dst_sub.mkdir(parents=True, exist_ok=True)
                for png_p in sub.glob("*.png"):
                    roi_map = cv2.imread(str(png_p), cv2.IMREAD_UNCHANGED).astype(np.float32)
                    coord_map = roi_map * spatial_weight
                    coord_uint16 = np.clip(coord_map, 0, 65535).astype(np.uint16)
                    cv2.imwrite(str(dst_sub / png_p.name), coord_uint16)

            count_c = len(list(dst_cat_test.glob("*/*.png")))
            print(f"  - {cat:16s} [ACTIVE]: {count_c} coordinate-weighted ROI maps exported")

    print("\n[SUCCESS] All 3 candidate configurations exported successfully!")


if __name__ == "__main__":
    export_all_configurations()
