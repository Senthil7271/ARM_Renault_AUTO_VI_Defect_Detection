"""AutoVI Edge Deployment: Stage 6 Prepare Hailo-8 Quantization Calibration Dataset.

Extracts representative normal (nominal) training images across all 6 AutoVI categories
and saves them as an uncompressed .npy file for Hailo Dataflow Compiler (DFC) post-training
quantization (PTQ).

Calibration Dataset:
- Format: numpy array uint8 [0, 255]
- Dimensions: (N, 256, 256, 3) (RGB)
- Source: Canonical AutoVI train/good normal samples only (zero test or defect data)
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

CATEGORIES = [
    "tank_screw",
    "pipe_clip",
    "engine_wiring",
    "pipe_staple",
    "underbody_pipes",
    "underbody_screw",
]

SAMPLES_PER_CATEGORY = 25  # Total = 150 calibration images (recommended Hailo PTQ range: 100-300)


def prepare_calibration_dataset():
    print("=" * 80)
    print("STAGE 6: PREPARING HAILO-8 CALIBRATION DATASET (PTQ)")
    print("=" * 80)

    calib_images = []
    manifest = []

    for cat in CATEGORIES:
        good_dir = WORKSPACE_ROOT / cat / cat / "train" / "good"
        if not good_dir.exists():
            print(f"[WARNING] Directory not found: {good_dir}")
            continue

        good_files = sorted(list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg")))
        selected = good_files[:SAMPLES_PER_CATEGORY]
        print(f"  - {cat:16s}: {len(selected)} / {len(good_files)} normal training samples selected")

        for img_path in selected:
            img = Image.open(img_path).convert("RGB")
            img_res = img.resize((256, 256), Image.BILINEAR)
            arr = np.array(img_res, dtype=np.uint8)  # (256, 256, 3) uint8
            calib_images.append(arr)
            manifest.append({
                "category": cat,
                "image_name": img_path.name,
                "path": str(img_path.resolve()),
                "shape": list(arr.shape),
                "dtype": "uint8",
            })

    calib_array = np.stack(calib_images, axis=0)  # (150, 256, 256, 3)
    out_dir = PROJECT_ROOT / "models" / "hailo"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npy = out_dir / "calibration_dataset.npy"
    np.save(str(out_npy), calib_array)

    manifest_json = out_dir / "calibration_manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": len(manifest),
            "array_shape": list(calib_array.shape),
            "dtype": str(calib_array.dtype),
            "source_provenance": "100% normal train/good training sets; zero test or defect exposure",
            "samples": manifest,
        }, f, indent=2)

    print(f"\n[SAVED] Calibration Dataset: {out_npy} ({calib_array.nbytes / (1024*1024):.2f} MB)")
    print(f"[SAVED] Calibration Manifest: {manifest_json}")
    print(f"Shape: {calib_array.shape}, Range: [{calib_array.min()}, {calib_array.max()}]")


if __name__ == "__main__":
    prepare_calibration_dataset()
