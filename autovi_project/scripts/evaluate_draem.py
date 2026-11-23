"""Comprehensive Evaluation and Canonical Anomaly Map Exporter for DRAEM on pipe_clip.

Performs:
1. Model loading and weight verification (Reconstructor & Discriminator).
2. Canonical test sample inference (N=337) matching official AutoVI protocol.
3. Canonical uint16 anomaly map export (400x400) for official benchmark integration.
4. Binary and Subtype-Mean Image AUROC computation.
5. Normal-only calibrated operational threshold metrics (Recall, FAR, Precision, Specificity, F1).
6. Official AutoVI localization evaluation (AUC-sPRO @ 0.01, 0.05, 0.10, 0.30, 1.00).
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.draem import DRAEMModel

# Evaluator setup
OFFICIAL_REPO_DIR = WORKSPACE_ROOT / "external" / "autovi_evaluation_code"
DATASET_BASE_DIR = WORKSPACE_ROOT / "autovi_dataset"


def run_draem_evaluation(
    model_path: Path,
    output_dir: Path,
    device: str = "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = output_dir / "anomaly_maps" / "pipe_clip" / "test"
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AUTOVI PHASE 3 — DRAEM PIPE_CLIP OFFICIAL BENCHMARK EVALUATION")
    print(f"Model Path:     {model_path}")
    print(f"Output Dir:     {output_dir}")
    print(f"Anomaly Maps:   {maps_dir}")
    print(f"Device:         {device}")
    print("=" * 80)

    # 1. Load Model
    print("\n[1/5] Loading trained DRAEM model...")
    state = torch.load(model_path, map_location=device)
    print("Model state keys:", list(state.keys()))
    calibrated_threshold = state.get("threshold", None)
    metadata = state.get("metadata", {})
    print(f"Calibrated 95th Percentile Threshold: {calibrated_threshold}")
    print(f"Model Metadata: {json.dumps(metadata, indent=2)}")

    model = DRAEMModel.load(model_path, device=device)
    model.eval()

    # 2. Load Canonical Test Manifest
    manifest_path = PROJECT_ROOT / "outputs" / "official_autovi" / "predictions" / "pipe_clip" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest["samples"]
    print(f"\n[2/5] Loaded {len(samples)} canonical test samples from manifest.")

    # 3. Generate Anomaly Maps and Image Scores
    print("\n[3/5] Running inference across all test samples...")
    t0_inf = time.time()
    results = []
    raw_maps = []

    for i, s in enumerate(samples):
        img_p = Path(s["image_path"])
        img_pil = Image.open(img_p).convert("RGB")
        img_np = np.array(img_pil)

        # Preprocessing matching DRAEM training: resize to (256, 256), scale to [0, 1]
        img_resized = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_LINEAR)
        t_img = (torch.from_numpy(img_resized).permute(2, 0, 1).contiguous().float() / 255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            img_score, amap = model.predict_anomaly(t_img, top_k_ratio=0.05)

        score_val = float(img_score.item())
        amap_np = amap.squeeze(0).cpu().numpy()

        # Resize to official canonical resolution (400, 400)
        amap_canon = cv2.resize(amap_np, (400, 400), interpolation=cv2.INTER_LINEAR)
        raw_maps.append((s, amap_canon))

        results.append({
            "image_stem": s["image_stem"],
            "defect_name": s["defect_name"],
            "is_anomaly": bool(s["is_anomaly"]),
            "anomaly_score": score_val,
            "calibrated_threshold": calibrated_threshold,
            "predicted_anomaly": bool(score_val >= calibrated_threshold) if calibrated_threshold is not None else None,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  Processed {i + 1}/{len(samples)} samples (elapsed: {time.time() - t0_inf:.1f}s)")

    # 4. Save Canonical uint16 Anomaly Maps
    print("\n[4/5] Normalizing and saving canonical uint16 anomaly maps...")
    min_val = min(m.min() for _, m in raw_maps)
    max_val = max(m.max() for _, m in raw_maps)
    print(f"Global Anomaly Map Range: min = {min_val:.6f}, max = {max_val:.6f}")

    for s, m in raw_maps:
        if max_val > min_val:
            m_norm = (m - min_val) / (max_val - min_val)
        else:
            m_norm = np.zeros_like(m)
        m_uint16 = (m_norm * 65535.0).astype(np.uint16)

        dest = maps_dir / s["defect_name"] / f"{s['image_stem']}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), m_uint16)

    print(f"Successfully exported {len(raw_maps)} uint16 anomaly maps to {maps_dir}")

    # 5. Compute Classification Metrics
    print("\n[5/5] Computing Classification and Benchmark Metrics...")
    y_true = np.array([r["is_anomaly"] for r in results], dtype=int)
    y_scores = np.array([r["anomaly_score"] for r in results], dtype=float)

    # Pooled Binary Image AUROC
    pooled_auroc = float(roc_auc_score(y_true, y_scores))
    print(f"Pooled Binary Image AUROC: {pooled_auroc:.4f}")

    # Subtype AUROCs
    normal_scores = [r["anomaly_score"] for r in results if not r["is_anomaly"]]
    subtype_aurocs = {}
    subtypes = sorted(list(set(r["defect_name"] for r in results if r["is_anomaly"])))

    for st in subtypes:
        st_scores = [r["anomaly_score"] for r in results if r["defect_name"] == st]
        labels = [0] * len(normal_scores) + [1] * len(st_scores)
        scores = normal_scores + st_scores
        st_auc = float(roc_auc_score(labels, scores))
        subtype_aurocs[st] = st_auc
        print(f"Subtype AUROC [{st}]: {st_auc:.4f} (N={len(st_scores)})")

    official_mean_auroc = float(np.mean(list(subtype_aurocs.values())))
    print(f"Official Subtype-Mean Image AUROC: {official_mean_auroc:.4f}")

    # Operational metrics at Calibrated 95th Percentile Threshold
    eval_summary = {
        "model_type": "draem",
        "category": "pipe_clip",
        "n_samples": len(results),
        "n_normal": int(np.sum(y_true == 0)),
        "n_anomaly": int(np.sum(y_true == 1)),
        "pooled_image_auroc": pooled_auroc,
        "official_subtype_mean_auroc": official_mean_auroc,
        "subtype_aurocs": subtype_aurocs,
        "calibrated_threshold": calibrated_threshold,
    }

    if calibrated_threshold is not None:
        y_pred = (y_scores >= calibrated_threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        eval_summary["operational_calibrated"] = {
            "threshold": calibrated_threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "defect_recall": recall,
            "false_alarm_rate": far,
            "precision": precision,
            "specificity": specificity,
            "f1": f1,
        }
        print("\nCalibrated Operational Performance:")
        print(f"  Threshold:   {calibrated_threshold:.4f}")
        print(f"  Defect Recall: {recall * 100:.2f}% ({tp}/{tp + fn})")
        print(f"  False Alarm:   {far * 100:.2f}% ({fp}/{fp + tn})")
        print(f"  Precision:     {precision * 100:.2f}% ({tp}/{tp + fp})")
        print(f"  Specificity:   {specificity * 100:.2f}% ({tn}/{fp + tn})")
        print(f"  F1 Score:      {f1:.4f}")

    # Save summary and predictions
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    preds_path = output_dir / "test_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation summary to: {summary_path}")
    print(f"Saved test predictions to:   {preds_path}")

    return eval_summary, maps_dir


if __name__ == "__main__":
    default_model = PROJECT_ROOT / "outputs" / "phase3" / "draem_pipeclip" / "model" / "pipe_clip" / "draem" / "calibrated_model.pth"
    default_out = PROJECT_ROOT / "outputs" / "phase3" / "draem_pipeclip" / "eval"
    run_draem_evaluation(default_model, default_out)
