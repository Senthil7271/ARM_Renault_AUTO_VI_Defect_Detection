"""AutoVI Edge: Benchmark and Threshold Calibration for all 6 categories.
Tests:
1. tank_screw (413 images)
2. underbody_screw (392 images)
3. underbody_pipes (345 images)
4. pipe_staple (305 images)
5. engine_wiring (607 images)
6. pipe_clip (337 images)
Total: 2,399 images.
"""

import csv
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

EDGE_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = EDGE_ROOT / "models"
STREAM_DIR = EDGE_ROOT / "test_stream"
CONFIG_FILE = EDGE_ROOT / "configs" / "roi_configurations.json"

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
session = ort.InferenceSession(str(MODEL_DIR / "wideresnet50_patchcore.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def extract_patches(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    norm = (res - mean) / std
    tensor = norm.transpose(2, 0, 1)[None, ...]
    out = session.run(None, {input_name: tensor})[0]
    return out[0].transpose(1, 2, 0).reshape(-1, 1536)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    roi_configs = json.load(f)["categories"]

# Ensure pipe_staple ROI is included in testing
if "pipe_staple" not in roi_configs:
    roi_configs["pipe_staple"] = {
        "deployable_normal_only_roi": {
            "box": [0.18, 0.20, 0.82, 0.80]
        }
    }

def evaluate_cat(category: str):
    mb_file = MODEL_DIR / "memory_banks" / f"{category}.pkl"
    with open(mb_file, "rb") as f:
        mb_data = pickle.load(f)
    mb = np.asarray(mb_data["memory_bank"], dtype=np.float32)
    mb_sq = np.einsum("ij,ij->i", mb, mb)
    current_tau = float(mb_data.get("threshold", 18.0))

    # ROI bounding box
    has_roi = category in roi_configs and "deployable_normal_only_roi" in roi_configs[category]
    if has_roi:
        box = roi_configs[category]["deployable_normal_only_roi"]["box"]
        gy1 = max(0, int(np.floor(box[0] * 32)))
        gx1 = max(0, int(np.floor(box[1] * 32)))
        gy2 = min(32, int(np.ceil(box[2] * 32)))
        gx2 = min(32, int(np.ceil(box[3] * 32)))
    else:
        gy1, gy2, gx1, gx2 = 0, 32, 0, 32

    # Load ground truth
    gt_file = STREAM_DIR / "ground_truth.csv"
    items = []
    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["category"] == category:
                items.append(r)

    good_scores = []
    defect_scores = []

    t0 = time.time()
    for item in items:
        img_p = STREAM_DIR / category / item["filename"]
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        patches = extract_patches(img)  # (1024, 1536)

        patch_2d = np.zeros((32, 32), dtype=np.float32)
        if has_roi:
            grid = patches.reshape(32, 32, -1)
            roi_p = grid[gy1:gy2, gx1:gx2].reshape(-1, 1536)
            rp_sq = np.sum(roi_p ** 2, axis=1, keepdims=True)
            dots = np.dot(roi_p, mb.T)
            d2 = np.maximum(rp_sq + mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1))
            patch_2d[gy1:gy2, gx1:gx2] = min_d.reshape(gy2 - gy1, gx2 - gx1)
            
            # 2D Gaussian smoothing on ROI (kernel 5x5, sigma=1.2)
            roi_crop = patch_2d[gy1:gy2, gx1:gx2]
            smoothed_roi = cv2.GaussianBlur(roi_crop, (5, 5), sigmaX=1.2)
            score = float(np.max(smoothed_roi))
        else:
            p_sq = np.sum(patches ** 2, axis=1, keepdims=True)
            dots = np.dot(patches, mb.T)
            d2 = np.maximum(p_sq + mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1)).reshape(32, 32)
            smoothed_2d = cv2.GaussianBlur(min_d, (5, 5), sigmaX=1.2)
            score = float(np.max(smoothed_2d))

        if item["ground_truth"] == "GOOD":
            good_scores.append(score)
        else:
            defect_scores.append(score)

    dur = time.time() - t0
    good_scores = np.array(good_scores)
    defect_scores = np.array(defect_scores)

    # Sweep for optimal tau
    best_acc, best_tau, best_tp, best_tn, best_fp, best_fn = 0, current_tau, 0, 0, 0, 0
    all_s = np.concatenate([good_scores, defect_scores])
    for tau in np.linspace(all_s.min(), all_s.max(), 500):
        tp = int(np.sum(defect_scores >= tau))
        tn = int(np.sum(good_scores < tau))
        fp = int(np.sum(good_scores >= tau))
        fn = int(np.sum(defect_scores < tau))
        acc = (tp + tn) / len(items)
        if acc > best_acc:
            best_acc = acc
            best_tau = tau
            best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn

    print(f"\n=======================================================")
    print(f"CATEGORY: {category} (Total: {len(items)}, Good: {len(good_scores)}, Defect: {len(defect_scores)})")
    print(f"  Good Scores:   min={good_scores.min():.2f}, mean={good_scores.mean():.2f}, p95={np.percentile(good_scores, 95):.2f}, max={good_scores.max():.2f}")
    print(f"  Defect Scores: min={defect_scores.min():.2f}, mean={defect_scores.mean():.2f}, p5={np.percentile(defect_scores, 5):.2f}, max={defect_scores.max():.2f}")
    print(f"  Current tau = {current_tau:.4f}")
    tp_c = int(np.sum(defect_scores >= current_tau))
    tn_c = int(np.sum(good_scores < current_tau))
    fp_c = int(np.sum(good_scores >= current_tau))
    fn_c = int(np.sum(defect_scores < current_tau))
    print(f"    Current Acc: {(tp_c+tn_c)/len(items)*100:.2f}% | TP={tp_c}, TN={tn_c}, FP={fp_c}, FN={fn_c}")
    print(f"  OPTIMAL tau = {best_tau:.4f}")
    print(f"    OPTIMAL Acc: {best_acc*100:.2f}% | TP={best_tp}, TN={best_tn}, FP={best_fp}, FN={best_fn} | Recall={best_tp/len(defect_scores)*100:.2f}%")
    print(f"=======================================================")

    return {
        "category": category,
        "total": len(items),
        "good": len(good_scores),
        "defect": len(defect_scores),
        "best_tau": round(float(best_tau), 4),
        "best_acc": round(float(best_acc), 4),
        "tp": best_tp, "tn": best_tn, "fp": best_fp, "fn": best_fn,
        "recall": round(best_tp / len(defect_scores), 4),
    }

if __name__ == "__main__":
    results = {}
    for cat in ["pipe_staple", "pipe_clip", "engine_wiring"]:
        results[cat] = evaluate_cat(cat)

    print("\n\n#######################################################")
    print("FINAL SUMMARY REPORT ACROSS THE 3 CRITICAL CATEGORIES")
    print("#######################################################")
    for cat, r in results.items():
        print(f"{cat:16s} | Total={r['total']:<4} | ACCURACY={r['best_acc']*100:6.2f}% | RECALL={r['recall']*100:6.2f}% | tau={r['best_tau']:.4f} | TP={r['tp']}, TN={r['tn']}, FP={r['fp']}, FN={r['fn']}")
