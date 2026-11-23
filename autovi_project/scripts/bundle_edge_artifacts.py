"""AutoVI Edge: Bundle Edge Artifacts and Serialization Script.

Copies and structures models, category memory banks, and ROI configurations into
the standalone, transfer-ready autovi-edge/ deployment package.
"""

import json
import os
import pickle
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDGE_ROOT = PROJECT_ROOT / "autovi-edge"

CATEGORIES = [
    "tank_screw",
    "pipe_clip",
    "engine_wiring",
    "pipe_staple",
    "underbody_pipes",
    "underbody_screw",
]


def bundle_edge_artifacts():
    print("=" * 80)
    print("BUNDLING STANDALONE AUTOVI-EDGE APPLICATION PACKAGE")
    print("=" * 80)

    # 1. Models directory
    models_out = EDGE_ROOT / "models"
    mb_out = models_out / "memory_banks"
    configs_out = EDGE_ROOT / "configs"

    models_out.mkdir(parents=True, exist_ok=True)
    mb_out.mkdir(parents=True, exist_ok=True)
    configs_out.mkdir(parents=True, exist_ok=True)

    # Copy ONNX Model
    src_onnx = PROJECT_ROOT / "models" / "onnx" / "wideresnet50_patchcore.onnx"
    dst_onnx = models_out / "wideresnet50_patchcore.onnx"
    if src_onnx.exists():
        shutil.copy2(src_onnx, dst_onnx)
        print(f"  [OK] Copied ONNX model: {dst_onnx.name} ({dst_onnx.stat().st_size / (1024*1024):.2f} MB)")

    # Copy HEF Model
    src_hef = PROJECT_ROOT / "models" / "hailo" / "wideresnet50_patchcore.hef"
    dst_hef = models_out / "wideresnet50_patchcore.hef"
    if src_hef.exists():
        shutil.copy2(src_hef, dst_hef)
        print(f"  [OK] Copied Hailo HEF model: {dst_hef.name}")

    # Copy / Serialize Memory Banks
    sagemaker_models = PROJECT_ROOT / "outputs" / "sagemaker_runs" / "autovi-all-2026-11-15-06-03-24" / "model"
    thresholds_dict = {}

    for cat in CATEGORIES:
        src_pkl = sagemaker_models / cat / "patchcore_model.pkl"
        dst_pkl = mb_out / f"{cat}.pkl"
        if src_pkl.exists():
            with open(src_pkl, "rb") as f:
                cat_data = pickle.load(f)
            
            mb_clean = {
                "category": cat,
                "memory_bank": cat_data["memory_bank"].astype(np.float32) if "np" in globals() else cat_data["memory_bank"],
                "threshold": float(cat_data["threshold"]),
                "min_score": 0.0,
                "max_score": float(cat_data["threshold"]) * 2.0,
            }
            with open(dst_pkl, "wb") as f:
                pickle.dump(mb_clean, f)
            print(f"  [OK] Serialized memory bank for '{cat}': shape {cat_data['memory_bank'].shape}, tau={cat_data['threshold']:.4f}")
            thresholds_dict[cat] = {
                "calibrated_threshold": round(float(cat_data["threshold"]), 4),
                "review_threshold": round(float(cat_data["threshold"]) * 0.85, 4),
                "memory_bank_coreset_size": cat_data["memory_bank"].shape[0],
                "feature_dimension": cat_data["memory_bank"].shape[1],
            }

    # Save operational thresholds config
    with open(configs_out / "operational_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"  [OK] Saved operational thresholds to: {configs_out / 'operational_thresholds.json'}")

    # Copy ROI configurations
    src_roi = PROJECT_ROOT / "configs" / "roi_configurations.json"
    dst_roi = configs_out / "roi_configurations.json"
    if src_roi.exists():
        shutil.copy2(src_roi, dst_roi)
        print(f"  [OK] Copied ROI configurations: {dst_roi.name}")

    print("\n[SUCCESS] Edge artifacts bundled successfully.")


if __name__ == "__main__":
    bundle_edge_artifacts()
