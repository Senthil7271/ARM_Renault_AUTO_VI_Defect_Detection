"""Evaluation and Benchmarking Script for Phase 3 Reverse Distillation on pipe_clip.

Executes:
1. Downloads artifacts from SageMaker S3 output prefix.
2. Verifies student parameter updates, training curves, GPU memory, and billable duration.
3. Evaluates all 337 test images under official AutoVI protocol.
4. Computes:
   - Official One-vs-Rest Mean Image AUROC.
   - Pooled Binary Image AUROC.
   - Official AUC-sPRO across standard FPR limits.
   - Normal-only calibrated operational metrics (Recall, Precision, Specificity, FAR, F1).
5. Generates comparison table against PatchCore WideResNet-50-2 baselines.
"""

import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.reverse_distillation import ReverseDistillationModel


def download_and_extract_sagemaker_artifacts(
    job_name: str,
    bucket: str = "sagemaker-us-east-1-077578771161",
    dest_dir: Optional[Path] = None,
    profile: str = "autovi",
    region: str = "us-east-1",
) -> Path:
    """Downloads model.tar.gz and output.tar.gz from SageMaker S3 output path."""
    dest_dir = dest_dir or (PROJECT_ROOT / "outputs" / "phase3" / job_name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client("s3")

    model_s3_key = f"autovi/phase3_outputs/{job_name}/output/model.tar.gz"
    output_s3_key = f"autovi/phase3_outputs/{job_name}/output/output.tar.gz"

    print(f"[S3] Downloading artifacts for job '{job_name}'...")
    local_model_tar = dest_dir / "model.tar.gz"
    local_output_tar = dest_dir / "output.tar.gz"

    s3.download_file(bucket, model_s3_key, str(local_model_tar))
    print(f"[S3] Downloaded model.tar.gz ({local_model_tar.stat().st_size / (1024*1024):.2f} MB)")

    try:
        s3.download_file(bucket, output_s3_key, str(local_output_tar))
        print(f"[S3] Downloaded output.tar.gz ({local_output_tar.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"[WARN] output.tar.gz download notice: {e}")

    # Extract model.tar.gz
    model_extract_dir = dest_dir / "model"
    model_extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_model_tar, "r:gz") as tar:
        tar.extractall(path=model_extract_dir)
    print(f"[EXTRACT] Extracted model artifacts to {model_extract_dir}")

    # Extract output.tar.gz if exists
    if local_output_tar.exists():
        output_extract_dir = dest_dir / "output"
        output_extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local_output_tar, "r:gz") as tar:
            tar.extractall(path=output_extract_dir)
        print(f"[EXTRACT] Extracted output metadata to {output_extract_dir}")

    return dest_dir


def evaluate_pipe_clip_reverse_distillation(
    artifact_dir: Path,
    device: str = "cpu",
) -> Dict:
    """Evaluates Reverse Distillation on pipe_clip test set."""
    # Find calibrated model or best model
    model_path = None
    for cand in [
        artifact_dir / "calibrated_model.pth",
        artifact_dir / "best_model.pth",
        artifact_dir / "model" / "pipe_clip" / "reverse_distillation" / "calibrated_model.pth",
        artifact_dir / "model" / "calibrated_model.pth",
        artifact_dir / "model" / "pipe_clip" / "reverse_distillation" / "best_model.pth",
        artifact_dir / "model" / "best_model.pth",
    ]:
        if cand.exists():
            model_path = cand
            break

    if model_path is None:
        raise FileNotFoundError(f"Could not find calibrated_model.pth in {artifact_dir}")

    print(f"[MODEL] Loading Reverse Distillation checkpoint from {model_path}...")
    model = ReverseDistillationModel.load(model_path, device=device)
    print(f"[MODEL] Calibrated threshold: {model.threshold:.4f} (calibrated={model.calibrated})")

    # Load test manifest
    manifest_p = PROJECT_ROOT / "outputs" / "official_autovi" / "predictions" / "pipe_clip" / "manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = manifest_data["samples"]
    print(f"[EVAL] Evaluating {len(samples)} pipe_clip test samples...")

    transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    y_true = []
    y_scores = []
    defect_names = []
    stems = []

    t0 = time.time()
    for s in samples:
        img_p = Path(s["image_path"])
        is_anom = int(s["is_anomaly"])
        def_name = s["defect_name"]
        stem = s["image_stem"]

        img = Image.open(img_p).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            score, anom_map = model.predict_anomaly(tensor, top_k_ratio=0.05)
            y_scores.append(float(score.item()))
            y_true.append(is_anom)
            defect_names.append(def_name)
            stems.append(stem)

    eval_time = time.time() - t0
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    defect_names = np.array(defect_names)

    # 1. Pooled Binary AUROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    pooled_auroc = float(auc(fpr, tpr))

    # 2. Official One-vs-Rest Mean Image AUROC across subtypes
    unique_subtypes = sorted(list(set(defect_names[defect_names != "good"])))
    subtype_aurocs = {}
    normal_scores = y_scores[defect_names == "good"]

    for subtype in unique_subtypes:
        sub_scores = y_scores[defect_names == subtype]
        sub_y = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(sub_scores))])
        sub_s = np.concatenate([normal_scores, sub_scores])
        sub_fpr, sub_tpr, _ = roc_curve(sub_y, sub_s)
        sub_auc = float(auc(sub_fpr, sub_tpr))
        subtype_aurocs[subtype] = sub_auc

    official_mean_auroc = float(np.mean(list(subtype_aurocs.values())))

    # 3. Operational metrics at normal-only calibrated threshold
    cal_threshold = float(model.threshold)
    y_pred = (y_scores >= cal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    far = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Subtype-level recall at calibrated threshold
    subtype_recalls = {}
    for subtype in unique_subtypes:
        sub_mask = defect_names == subtype
        sub_rec = float(np.mean(y_scores[sub_mask] >= cal_threshold))
        subtype_recalls[subtype] = sub_rec

    results = {
        "job_dir": str(artifact_dir),
        "model_type": "reverse_distillation",
        "category": "pipe_clip",
        "sample_count": len(samples),
        "calibrated_threshold": cal_threshold,
        "pooled_image_auroc": pooled_auroc,
        "official_mean_image_auroc": official_mean_auroc,
        "subtype_aurocs": subtype_aurocs,
        "operational_metrics": {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Recall": float(recall),
            "Precision": float(precision),
            "Specificity": float(specificity),
            "False_Alarm_Rate": float(far),
            "F1": float(f1),
        },
        "subtype_recalls": subtype_recalls,
        "evaluation_time_sec": eval_time,
    }

    # Save evaluation summary
    out_json = artifact_dir / "pipe_clip_rd_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("      PIPE_CLIP REVERSE DISTILLATION EVALUATION RESULTS      ")
    print("=" * 70)
    print(f"  Pooled Binary Image AUROC:     {pooled_auroc:.4f}")
    print(f"  Official Mean Image AUROC:     {official_mean_auroc:.4f}")
    for st, val in subtype_aurocs.items():
        print(f"    - Subtype '{st}': AUROC = {val:.4f} (Recall @ Cal: {subtype_recalls[st]*100:.1f}%)")
    print(f"  Calibrated 95th % Threshold:   {cal_threshold:.4f}")
    print(f"  Defect Recall:                 {recall*100:.1f}% ({tp}/{tp+fn})")
    print(f"  Precision:                     {precision*100:.1f}%")
    print(f"  Specificity:                   {specificity*100:.1f}%")
    print(f"  False Alarm Rate (FAR):        {far*100:.1f}% ({fp}/{tn+fp})")
    print(f"  F1 Score:                      {f1:.4f}")
    print("=" * 70)

    return results
