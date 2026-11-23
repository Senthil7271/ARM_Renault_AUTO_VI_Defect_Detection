#!/usr/bin/env python3
"""AutoVI Edge: Dedicated Native PyTorch FP32 Test & Evaluation Script.

Runs WideResNet-50-2 PatchCore directly in pure PyTorch (torch.float32) on Arm CPU.
Uses Config C (Deployable Normal ROI + Spatial Coordinate Conditioning + Gaussian Blur).

Usage:
  python3 evaluate_pytorch_cpu.py --category pipe_clip
  python3 evaluate_pytorch_cpu.py --category pipe_staple
  python3 evaluate_pytorch_cpu.py --category engine_wiring
"""

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

ROOT = Path(__file__).resolve().parent
MB_DIR = ROOT / "models" / "memory_banks"
CONFIG_FILE = ROOT / "configs" / "roi_configurations.json"
STREAM_DIR = ROOT / "test_stream"
GT_FILE = STREAM_DIR / "ground_truth.csv"

# ImageNet normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WideResNet50PatchCore(nn.Module):
    """Native PyTorch WideResNet-50-2 PatchCore Feature Extractor (FP32)."""

    def __init__(self):
        super().__init__()
        bb = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.conv1 = bb.conv1
        self.bn1 = bb.bn1
        self.relu = bb.relu
        self.maxpool = bb.maxpool
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        f2 = self.layer2(x)      # (B, 512, 32, 32)
        f3 = self.layer3(f2)     # (B, 1024, 16, 16)

        p2 = self.avg_pool(f2)   # (B, 512, 32, 32)
        p3 = self.avg_pool(f3)   # (B, 1024, 16, 16)

        p3_up = F.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)
        return torch.cat([p2, p3_up], dim=1)  # (B, 1536, 32, 32)


class PyTorchAnomalyEvaluator:
    def __init__(self, num_threads: int = 4):
        print("Initializing Native PyTorch FP32 Backbone...")
        torch.set_num_threads(num_threads)
        self.device = torch.device("cpu")
        self.model = WideResNet50PatchCore().to(self.device).eval()
        print(f"PyTorch FP32 Initialized on CPU ({num_threads} threads).")

        # Load ROI configurations
        self.roi_configs = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self.roi_configs = json.load(f).get("categories", {})

        self.banks = {}
        self.active_cat = None

    def set_category(self, category: str):
        if category in self.banks:
            self.active_cat = category
            return

        pkl_path = MB_DIR / f"{category}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Memory bank not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        mb = np.asarray(data["memory_bank"], dtype=np.float32)
        mb_sq = np.einsum("ij,ij->i", mb, mb)
        threshold = float(data.get("threshold", 17.50))

        # Setup Config C Deployable ROI + Coordinate Conditioning
        roi_meta = None
        if category in self.roi_configs:
            box = self.roi_configs[category]["deployable_normal_only_roi"]["box"]
            gy1, gx1 = max(0, int(np.floor(box[0] * 32))), max(0, int(np.floor(box[1] * 32)))
            gy2, gx2 = min(32, int(np.ceil(box[2] * 32))), min(32, int(np.ceil(box[3] * 32)))

            cy, cx = (gy1 + gy2) / 2.0, (gx1 + gx2) / 2.0
            sy, sx = max(1.0, (gy2 - gy1) / 2.0), max(1.0, (gx2 - gx1) / 2.0)
            yy, xx = np.mgrid[0:32, 0:32]
            dist_sq = ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2
            spatial_weight = np.exp(-0.2 * dist_sq).astype(np.float32)

            roi_meta = {
                "gy1": gy1, "gy2": gy2, "gx1": gx1, "gx2": gx2,
                "spatial_weight": spatial_weight,
            }

        self.banks[category] = {
            "mb": mb,
            "mb_sq": mb_sq,
            "threshold": threshold,
            "roi": roi_meta,
        }
        self.active_cat = category
        print(f"Loaded memory bank for '{category}': shape {mb.shape}, tau={threshold:.4f}")

    def inspect(self, img_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()

        # 1. Preprocess
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        norm = (img_res - MEAN) / STD

        # 2. PyTorch FP32 Backbone Feature Extraction
        with torch.inference_mode():
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)[None, ...]).float()
            out = self.model(tensor).cpu().numpy()
            # out shape: (1, 1536, 32, 32)
            patches_grid = out[0].transpose(1, 2, 0)

        # 3. Nearest-Neighbor Matching
        cat_info = self.banks[self.active_cat]
        mb, mb_sq, roi, tau = cat_info["mb"], cat_info["mb_sq"], cat_info["roi"], cat_info["threshold"]

        if roi is not None:
            gy1, gy2, gx1, gx2 = roi["gy1"], roi["gy2"], roi["gx1"], roi["gx2"]
            roi_patches = patches_grid[gy1:gy2, gx1:gx2].reshape(-1, 1536)

            p_sq = np.sum(roi_patches ** 2, axis=1, keepdims=True)
            dots = np.dot(roi_patches, mb.T)
            d2 = np.maximum(p_sq + mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1))

            roi_dists = min_d.reshape(gy2 - gy1, gx2 - gx1)
            sw = roi["spatial_weight"][gy1:gy2, gx1:gx2]
            weighted_roi = roi_dists * sw

            smoothed_roi = cv2.GaussianBlur(weighted_roi, (5, 5), sigmaX=1.2)
            score = float(np.max(smoothed_roi))
        else:
            full_patches = patches_grid.reshape(-1, 1536)
            p_sq = np.sum(full_patches ** 2, axis=1, keepdims=True)
            dots = np.dot(full_patches, mb.T)
            d2 = np.maximum(p_sq + mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1))
            dist_2d = min_d.reshape(32, 32)
            smoothed_2d = cv2.GaussianBlur(dist_2d, (5, 5), sigmaX=1.2)
            score = float(np.max(smoothed_2d))

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "score": score,
            "threshold": tau,
            "is_reject": score >= tau,
            "latency_ms": round(total_ms, 1),
        }


def main():
    parser = argparse.ArgumentParser(description="AutoVI Dedicated Native PyTorch FP32 Evaluator")
    parser.add_argument("--category", default="pipe_clip", help="Category name or 'all'")
    parser.add_argument("--csv", default="", help="Output CSV filename")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch")
    args = parser.parse_args()

    if not GT_FILE.exists():
        sys.exit(f"[ERROR] Ground truth file not found at: {GT_FILE}")

    items = []
    with open(GT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if args.category == "all" or r["category"] == args.category:
                items.append(r)

    if not items:
        sys.exit(f"[ERROR] No images found for category '{args.category}'.")

    if args.limit > 0:
        items = items[: args.limit]

    out_csv = args.csv if args.csv else f"{args.category}_pytorch_fp32_eval.csv"

    print("=" * 105)
    print(" AUTOVI INDUSTRIAL EVALUATION AUDIT (NATIVE PYTORCH FP32 CPU)")
    print(f" Target: {args.category} | Model: WideResNet-50-2 PatchCore FP32 | Threads: {args.threads}")
    print(f" Images: {len(items)} (Good: {sum(1 for x in items if x['ground_truth'] == 'GOOD')}, Defect: {sum(1 for x in items if x['ground_truth'] == 'DEFECT')})")
    print("=" * 105)

    evaluator = PyTorchAnomalyEvaluator(num_threads=args.threads)

    csv_file = open(out_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "category", "filename", "ground_truth", "defect_type", "anomaly_score", "threshold", "verdict", "evaluation_result", "latency_ms"])

    print(f"\n{'FRAME':<6} | {'CATEGORY':<14} | {'FILE':<16} | {'GT':<7} | {'SCORE':<7} | {'TAU':<6} | {'VERDICT':<10} | {'RESULT':<12} | {'LATENCY':<9}")
    print("-" * 105)

    cat_stats = {}
    current_cat = None

    for idx, item in enumerate(items, start=1):
        cat = item["category"]
        rel_path = item["filename"]
        gt = item["ground_truth"]
        defect_type = item.get("defect_type", "")

        if cat not in cat_stats:
            cat_stats[cat] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0, "time_ms": 0.0, "good": 0, "defect": 0}
        cat_stats[cat]["total"] += 1
        if gt == "DEFECT":
            cat_stats[cat]["defect"] += 1
        else:
            cat_stats[cat]["good"] += 1

        img_p = STREAM_DIR / cat / rel_path
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        if cat != current_cat:
            evaluator.set_category(cat)
            current_cat = cat

        res = evaluator.inspect(img)
        score, tau, lat, is_reject = res["score"], res["threshold"], res["latency_ms"], res["is_reject"]
        cat_stats[cat]["time_ms"] += lat
        verdict = "REJECT" if is_reject else "PASS"

        if gt == "DEFECT":
            if is_reject:
                cat_stats[cat]["tp"] += 1
                res_str = "\033[92mMATCH [TP]\033[0m"
                eval_res = "CORRECT (TP)"
            else:
                cat_stats[cat]["fn"] += 1
                res_str = "\033[91mESCAPE [FN]\033[0m"
                eval_res = "ESCAPE (FN)"
        else:
            if not is_reject:
                cat_stats[cat]["tn"] += 1
                res_str = "\033[92mMATCH [TN]\033[0m"
                eval_res = "CORRECT (TN)"
            else:
                cat_stats[cat]["fp"] += 1
                res_str = "\033[91mFALSE [FP]\033[0m"
                eval_res = "FALSE_ALARM (FP)"

        v_str = "\033[91mREJECT   \033[0m" if is_reject else "\033[92mPASS [OK]\033[0m"
        short_file = rel_path if len(rel_path) <= 16 else rel_path[-16:]

        print(f"#{idx:<5} | {cat:<14} | {short_file:<16} | {gt:<7} | {score:<7.3f} | {tau:<6.2f} | {v_str} | {res_str} | {lat:6.1f} ms", flush=True)

        csv_writer.writerow([idx, cat, rel_path, gt, defect_type, round(score, 4), round(tau, 4), verdict, eval_res, lat])

    csv_file.close()

    print("\n" + "=" * 115)
    print(" AUTOVI INDUSTRIAL EVALUATION SUMMARY AUDIT (NATIVE PYTORCH FP32 CPU)")
    print("=" * 115)
    print(f"{'CATEGORY':<18} | {'TOTAL':<6} | {'GOOD':<5} | {'DEFECT':<6} | {'TP':<4} | {'TN':<4} | {'FP':<4} | {'FN':<4} | {'ACCURACY':<9} | {'RECALL':<8} | {'AVG LAT':<8}")
    print("-" * 115)

    tot_tp, tot_tn, tot_fp, tot_fn, tot_all, tot_time = 0, 0, 0, 0, 0, 0.0
    for c_name, st in sorted(cat_stats.items()):
        acc = ((st["tp"] + st["tn"]) / st["total"]) * 100.0 if st["total"] > 0 else 0.0
        rec = (st["tp"] / st["defect"]) * 100.0 if st["defect"] > 0 else 0.0
        avg_l = st["time_ms"] / st["total"] if st["total"] > 0 else 0.0
        print(f"{c_name:<18} | {st['total']:<6} | {st['good']:<5} | {st['defect']:<6} | {st['tp']:<4} | {st['tn']:<4} | {st['fp']:<4} | {st['fn']:<4} | {acc:8.2f}% | {rec:7.2f}% | {avg_l:6.1f} ms")
        tot_tp += st["tp"]
        tot_tn += st["tn"]
        tot_fp += st["fp"]
        tot_fn += st["fn"]
        tot_all += st["total"]
        tot_time += st["time_ms"]

    print("-" * 115)
    o_acc = ((tot_tp + tot_tn) / tot_all) * 100.0 if tot_all > 0 else 0.0
    o_rec = (tot_tp / (tot_tp + tot_fn)) * 100.0 if (tot_tp + tot_fn) > 0 else 0.0
    o_lat = tot_time / tot_all if tot_all > 0 else 0.0
    print(f"{'OVERALL':<18} | {tot_all:<6} | {tot_all - (tot_tp + tot_fn):<5} | {tot_tp + tot_fn:<6} | {tot_tp:<4} | {tot_tn:<4} | {tot_fp:<4} | {tot_fn:<4} | {o_acc:8.2f}% | {o_rec:7.2f}% | {o_lat:6.1f} ms")
    print("=" * 115)
    print(f" Detailed Audit CSV Saved To: {out_csv}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
