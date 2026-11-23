"""Exports test anomaly maps and predictions from trained PatchCore models

Formats outputs to satisfy the official AutoVI benchmark evaluation contract:
<anomaly_maps_dir>/<object_name>/test/<defect_name>/<image_id>.png

Saves:
1. 16-bit uint16 PNG anomaly maps at canonical evaluation resolutions.
2. Raw float32 anomaly maps (.npy) for full-precision statistical verification.
3. Comprehensive prediction manifests (predictions.csv) per category.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchcore_model import PatchCoreModel


CANONICAL_RESIZE_SHAPES = {
    "pipe_clip": (400, 400),
    "pipe_staple": (400, 400),
    "engine_wiring": (400, 400),
    "tank_screw": (1000, 750),
    "underbody_pipes": (1000, 750),
    "underbody_screw": (1000, 750),
}


def get_category_test_samples(dataset_root: Path, category: str) -> List[Dict]:
    """Discovers all test samples for a category, maintaining official folder structure."""
    cat_dir = dataset_root / category / category / "test"
    if not cat_dir.exists():
        # Try single-nested
        cat_dir = dataset_root / category / "test"

    samples = []
    for defect_folder in sorted(cat_dir.iterdir()):
        if not defect_folder.is_dir():
            continue
        defect_name = defect_folder.name
        is_anomaly = 0 if defect_name == "good" else 1

        for img_p in sorted(defect_folder.glob("*.png")):
            samples.append({
                "image_path": img_p,
                "defect_name": defect_name,
                "image_stem": img_p.stem,
                "is_anomaly": is_anomaly,
                "rel_path": f"{defect_name}/{img_p.name}",
            })

    return samples


def export_category_predictions(
    category: str,
    model_checkpoint_path: Path,
    dataset_root: Path,
    output_base_dir: Path,
    batch_size: int = 16,
    device: str = "cpu",
):
    print(f"\n=======================================================")
    print(f"Exporting Official Predictions: {category}")
    print(f"Checkpoint: {model_checkpoint_path}")
    print(f"=======================================================")

    target_shape = CANONICAL_RESIZE_SHAPES.get(category, (400, 400))
    print(f"Target Evaluation Resolution (W, H): {target_shape}")

    # Load Model
    t0_load = time.perf_counter()
    model = PatchCoreModel.load(model_checkpoint_path, device=device)
    load_time = time.perf_counter() - t0_load
    print(f"Model loaded in {load_time:.2f}s on {device}.")
    print(f"Memory Bank Shape: {model.memory_bank.shape}, Calibrated Threshold: {model.threshold:.4f}")

    # Prepare Transforms
    transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Discover Test Samples
    samples = get_category_test_samples(dataset_root, category)
    print(f"Found {len(samples)} test samples.")

    # Destination directories
    # Official structure: <output_base_dir>/anomaly_maps/<category>/test/<defect_name>/<image_id>.png
    anomaly_maps_cat_dir = output_base_dir / "anomaly_maps" / category / "test"
    raw_scores_cat_dir = output_base_dir / "predictions" / category / "raw_scores"

    anomaly_maps_cat_dir.mkdir(parents=True, exist_ok=True)
    raw_scores_cat_dir.mkdir(parents=True, exist_ok=True)

    # First Pass: Generate all raw anomaly maps in batches
    raw_maps = []
    image_scores = []
    manifest_rows = []

    t0_inf = time.perf_counter()
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        tensors = []
        for s in batch_samples:
            img = Image.open(s["image_path"]).convert("RGB")
            tensors.append(transform(img))
        batch_tensor = torch.stack(tensors, dim=0).to(model.device)

        with torch.inference_mode():
            scores, heatmaps = model.predict_batch(batch_tensor)

        for s, score, hm in zip(batch_samples, scores, heatmaps):
            if isinstance(hm, torch.Tensor):
                hm_np = hm.squeeze().cpu().numpy().astype(np.float32)
            else:
                hm_np = np.squeeze(hm).astype(np.float32)
            raw_maps.append(hm_np)
            image_scores.append(float(score))
            manifest_rows.append({
                "image_path": str(s["image_path"]),
                "defect_name": s["defect_name"],
                "image_stem": s["image_stem"],
                "is_anomaly": s["is_anomaly"],
                "anomaly_score": float(score),
                "threshold": float(model.threshold),
                "prediction": "anomaly" if float(score) >= model.threshold else "normal",
                "map_min": float(hm_np.min()),
                "map_max": float(hm_np.max()),
            })

    total_inf_time = time.perf_counter() - t0_inf
    print(f"Inference completed in {total_inf_time:.2f}s ({total_inf_time / len(samples):.4f}s per image).")

    # Global min/max normalization across the category for 16-bit PNG export
    all_min = min(r["map_min"] for r in manifest_rows)
    all_max = max(r["map_max"] for r in manifest_rows)
    print(f"Category Global Score Dynamic Range: [{all_min:.4f}, {all_max:.4f}]")

    # Second Pass: Resize to canonical shape and save
    print(f"Saving anomaly maps to {anomaly_maps_cat_dir}...")
    for s, raw_map, row in zip(samples, raw_maps, manifest_rows):
        # Resize anomaly map to canonical resolution (target_shape is (W, H))
        # cv2.resize expects (W, H)
        resized_raw = cv2.resize(raw_map, target_shape, interpolation=cv2.INTER_LINEAR)

        # 1. Save raw float32 .npy
        defect_raw_dir = raw_scores_cat_dir / s["defect_name"]
        defect_raw_dir.mkdir(parents=True, exist_ok=True)
        raw_npy_path = defect_raw_dir / f"{s['image_stem']}.npy"
        np.save(str(raw_npy_path), resized_raw)

        # 2. Convert to 16-bit uint16 PNG using global dynamic range
        # uint16 range: 0 .. 65535
        if all_max > all_min:
            norm_map = (resized_raw - all_min) / (all_max - all_min)
            uint16_map = np.clip(norm_map * 65535.0, 0, 65535).astype(np.uint16)
        else:
            uint16_map = np.zeros(resized_raw.shape, dtype=np.uint16)

        defect_png_dir = anomaly_maps_cat_dir / s["defect_name"]
        defect_png_dir.mkdir(parents=True, exist_ok=True)
        png_path = defect_png_dir / f"{s['image_stem']}.png"
        cv2.imwrite(str(png_path), uint16_map)

    # Save Category Manifest
    manifest_file = output_base_dir / "predictions" / category / "manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "category": category,
            "num_samples": len(samples),
            "canonical_resolution": target_shape,
            "global_min_score": all_min,
            "global_max_score": all_max,
            "load_time_seconds": load_time,
            "inference_time_seconds": total_inf_time,
            "checkpoint_path": str(model_checkpoint_path),
            "samples": manifest_rows,
        }, f, indent=2)

    print(f"[DONE] Category {category} exported successfully. Manifest: {manifest_file}")


def main():
    parser = argparse.ArgumentParser(description="Export AutoVI predictions for official benchmark")
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Category name or 'all' for all 6 categories",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "sagemaker_runs" / "autovi-all-2026-11-15-06-03-24" / "model"),
        help="Directory containing trained model checkpoints",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(PROJECT_ROOT.parent),
        help="Workspace root containing AutoVI category folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "official_autovi"),
        help="Directory to save official anomaly maps and predictions",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for feature extraction")
    args = parser.parse_args()

    categories = [
        "pipe_clip",
        "pipe_staple",
        "engine_wiring",
        "tank_screw",
        "underbody_pipes",
        "underbody_screw",
    ] if args.category.lower() == "all" else [args.category]

    ckpt_base = Path(args.checkpoint_dir).resolve()
    ds_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.output_dir).resolve()

    for cat in categories:
        ckpt_p = ckpt_base / cat / "patchcore_model.pkl"
        if not ckpt_p.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_p}")
            continue
        export_category_predictions(
            category=cat,
            model_checkpoint_path=ckpt_p,
            dataset_root=ds_root,
            output_base_dir=out_dir,
            batch_size=args.batch_size,
            device="cpu",
        )


if __name__ == "__main__":
    main()
