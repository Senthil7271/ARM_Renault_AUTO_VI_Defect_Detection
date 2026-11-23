"""SageMaker Training Entry Point for AutoVI Phase 3 Deep Anomaly Models.

Executes inside SageMaker PyTorch Deep Learning Container on GPU (ml.g4dn.xlarge, NVIDIA Tesla T4).
Supports:
1. Reverse Distillation (Frozen Pretrained Teacher + Trainable Student Decoder).
2. DRAEM (Reconstructive + Discriminative Subnetworks on procedural pseudo-anomalies).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Defense-in-depth: Ensure NumPy 1.x compatibility with PyTorch 2.1.0 DLC (Container only)
is_sagemaker = any(k in os.environ for k in ("SM_TRAINING_ENV", "SM_MODEL_DIR", "SM_CHANNEL_TRAIN")) or Path("/opt/ml").exists()
if is_sagemaker:
    try:
        import numpy as _np
        _np_major = int(_np.__version__.split(".")[0])
        if _np_major >= 2:
            print(f"[RECOVERY] Incompatible NumPy {_np.__version__} detected in container.")
            print(f"[RECOVERY] Downgrading to numpy==1.26.4 and opencv-python-headless==4.9.0.80...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "numpy==1.26.4", "opencv-python-headless==4.9.0.80", "scipy<1.14.0"
            ])
            print(f"[RECOVERY] Environment repaired. Relaunching entrypoint...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as _err:
        pass

# Add project root and source tree to sys.path
CODE_DIR = Path(__file__).resolve().parent
if not (CODE_DIR / "src").exists() and (CODE_DIR.parent / "src").exists():
    CODE_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from src.models.draem import DRAEMModel
from src.models.reverse_distillation import ReverseDistillationModel
from src.training.train_deep import train_draem, train_reverse_distillation


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="AutoVI Phase 3 Deep Anomaly Training Entry Point")

    # SageMaker standard path arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Root directory containing AutoVI categories",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
        help="Directory to save evaluation reports and training history",
    )

    # Architecture and training hyperparameters
    parser.add_argument(
        "--model-type",
        type=str,
        default="reverse_distillation",
        choices=["reverse_distillation", "draem"],
        help="Model architecture: reverse_distillation or draem",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="pipe_clip",
        choices=["pipe_clip", "tank_screw", "engine_wiring"],
        help="AutoVI target priority category",
    )
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="Teacher backbone for RD")
    parser.add_argument("--bottleneck-dim", type=int, default=512, help="Bottleneck feature dimension for RD")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    return parser.parse_args(args)


def resolve_category_train_val_paths(data_dir: Path, category: str, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """Resolves normal training and validation image paths from standard dataset directory."""
    # Check double-nested path vs flat path
    train_good_dir = data_dir / category / category / "train" / "good"
    if not train_good_dir.exists():
        train_good_dir = data_dir / category / "train" / "good"
        if not train_good_dir.exists():
            train_good_dir = data_dir / "train" / "good"

    if not train_good_dir.exists():
        raise FileNotFoundError(f"Could not locate train/good directory for category '{category}' in '{data_dir}'")

    all_good_images = sorted(list(train_good_dir.glob("*.png")) + list(train_good_dir.glob("*.jpg")))
    if len(all_good_images) == 0:
        raise ValueError(f"No normal training images found in {train_good_dir}")

    # Check for split manifest in code tree or data dir
    manifest_candidates = [
        CODE_DIR / "outputs" / "sagemaker_runs" / "autovi-all-2026-11-15-06-03-24" / "output" / category / "manifests" / f"{category}_split_manifest.json",
        CODE_DIR / "manifests" / f"{category}_split_manifest.json",
        data_dir / f"{category}_split_manifest.json",
    ]
    manifest_path = None
    for mc in manifest_candidates:
        if mc.exists():
            manifest_path = mc
            break

    if manifest_path is not None:
        print(f"[DATA] Reusing validated split manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_paths = [train_good_dir / fname for fname in manifest["train_images"] if (train_good_dir / fname).exists()]
        val_paths = [train_good_dir / fname for fname in manifest["val_images"] if (train_good_dir / fname).exists()]
    else:
        print(f"[DATA] Generating 90/10 split on {len(all_good_images)} normal images (seed={seed})...")
        rng = np.random.default_rng(seed)
        shuffled = list(all_good_images)
        rng.shuffle(shuffled)
        n_val = max(15, int(len(shuffled) * 0.10))
        val_paths = sorted(shuffled[:n_val], key=lambda p: p.name)
        train_paths = sorted(shuffled[n_val:], key=lambda p: p.name)

    print(f"[DATA] Category: {category} | Train Normal: {len(train_paths)} | Val Normal: {len(val_paths)}")
    return train_paths, val_paths


def main():
    args = parse_args()

    print("=" * 80)
    print("AUTOVI PHASE 3 — AWS SAGEMAKER DEEP ANOMALY TRAINING")
    print(f"Model Type:    {args.model_type.upper()}")
    print(f"Category:      {args.category}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Device:        {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU Device:    {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir) / args.category / args.model_type
    output_dir = Path(args.output_dir) / args.category / args.model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Dataset Paths
    train_paths, val_paths = resolve_category_train_val_paths(data_dir, args.category, seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 2. Execute Training
    t0 = time.time()
    if args.model_type == "reverse_distillation":
        model, summary = train_reverse_distillation(
            train_paths=train_paths,
            val_paths=val_paths,
            output_dir=model_dir,
            backbone_name=args.backbone,
            bottleneck_dim=args.bottleneck_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            use_amp=True,
            seed=args.seed,
            patience=args.patience,
        )
    elif args.model_type == "draem":
        model, summary = train_draem(
            train_paths=train_paths,
            val_paths=val_paths,
            output_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            use_amp=True,
            seed=args.seed,
            patience=args.patience,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    total_training_time = time.time() - t0
    peak_vram_mb = 0.0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    # 3. Normal-Only Threshold Calibration (Target FPR = 0.05 on held-out validation)
    print("\n[CALIBRATION] Calibrating 95th percentile threshold on held-out nominals...")
    model.eval()
    val_scores = []
    with torch.no_grad():
        for vp in val_paths:
            img = cv2.imread(str(vp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            if args.model_type == "reverse_distillation":
                t_img = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])(img).unsqueeze(0).to(device)
            else:
                t_img = (torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0).unsqueeze(0).to(device)

            score, _ = model.predict_anomaly(t_img)
            val_scores.append(float(score.item()))

    val_scores = np.sort(np.array(val_scores))
    calibrated_q95 = float(np.quantile(val_scores, 0.95))
    model.threshold = calibrated_q95
    model.calibrated = True

    # Save calibrated model
    calib_model_path = model_dir / "calibrated_model.pth"
    model.save(calib_model_path, metadata={"calibrated_threshold": calibrated_q95, "n_val": len(val_scores)})
    print(f"[CALIBRATION] Calibrated 95th percentile threshold: {calibrated_q95:.4f} (N={len(val_scores)})")

    # 4. Save Complete Training & System Metadata
    training_meta = {
        "project": "AutoVI_Phase_3",
        "category": args.category,
        "model_type": args.model_type,
        "backbone": args.backbone if args.model_type == "reverse_distillation" else "unet",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "peak_vram_mib": peak_vram_mb,
        "training_time_sec": total_training_time,
        "calibrated_threshold": calibrated_q95,
        "val_score_mean": float(np.mean(val_scores)),
        "val_score_std": float(np.std(val_scores)),
        "summary": summary,
    }

    meta_path = output_dir / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    print("\n[COMPLETE] Phase 3 Training Job Completed Successfully!")
    print(f"  Artifacts saved to: {model_dir}")
    print(f"  Metadata saved to:  {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
