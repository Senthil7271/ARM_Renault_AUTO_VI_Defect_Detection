#!/usr/bin/env python3
"""AutoVI Edge: Complete Industrial Dataset Stream Evaluation & CSV Logger.

Streams real test images (mixing nominal good parts and defects in realistic sequence),
performs end-to-end PatchCore inference, evaluates verdicts against ground truth,
and logs all per-frame telemetry to an industrial audit CSV.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EDGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EDGE_ROOT))

from src.decision_engine import DecisionEngine, Verdict
from src.inference_engine import EdgeAnomalyEngine


def main():
    parser = argparse.ArgumentParser(description="AutoVI Dataset Stream Evaluator")
    parser.add_argument("--category", default="tank_screw", help="Specific category or 'all'")
    parser.add_argument("--backend", default="auto", choices=["auto", "hailo", "onnx"])
    parser.add_argument("--device", default="auto", help="Alias for backend")
    parser.add_argument("--csv", default="inspection_results.csv", help="Output CSV path")
    parser.add_argument("--summary-csv", default="inspection_summary.csv", help="Summary CSV path")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Mix good and defect images randomly")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--fast", action="store_true", default=False, help="Use subsampled bank for fast throughput")
    parser.add_argument("--delay", type=float, default=0.01, help="Inter-frame delay in seconds to prevent thermal throttling")
    args = parser.parse_args()

    backend = args.backend if args.backend != "auto" else args.device
    stream_dir = EDGE_ROOT / "test_stream"
    gt_file = stream_dir / "ground_truth.csv"

    if not gt_file.exists():
        sys.exit(f"[ERROR] Ground truth manifest not found at: {gt_file}\nPlease ensure test_stream is extracted.")

    print("=" * 95)
    print(" AUTOVI EDGE: INDUSTRIAL REAL-TIME TEST STREAM EVALUATOR")
    print(f" Category: {args.category} | Backend: {backend} | Output CSV: {args.csv}")
    print("=" * 95)

    # 1. Load Ground Truth Items
    items = []
    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.category == "all" or row["category"] == args.category:
                items.append(row)

    if not items:
        sys.exit(f"[ERROR] No test images found for category '{args.category}'.")

    if args.shuffle:
        random.seed(42)
        random.shuffle(items)

    if args.limit > 0:
        items = items[:args.limit]

    print(f"Loaded {len(items)} test images for evaluation (Good: {sum(1 for x in items if x['ground_truth'] == 'GOOD')}, Defect: {sum(1 for x in items if x['ground_truth'] == 'DEFECT')}).\n")

    # 2. Initialize Engines
    model_dir = EDGE_ROOT / "models"
    print("Initializing inference engine...")
    t0 = time.perf_counter()
    engine = EdgeAnomalyEngine(model_dir=model_dir, preferred_backend=backend)
    decision_engine = DecisionEngine()
    print(f"Engine Ready | Active Backend: {engine.backend_name}\n")

    # 3. Setup CSV output
    csv_path = Path(args.csv)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "category", "filename", "ground_truth", "defect_type",
        "anomaly_score", "threshold", "verdict", "evaluation_result", "latency_ms"
    ])

    print(f"{'FRAME':<6} | {'CATEGORY':<14} | {'FILE':<16} | {'GT':<7} | {'SCORE':<7} | {'TAU':<6} | {'VERDICT':<10} | {'RESULT':<12} | {'LATENCY':<9}")
    print("-" * 105)

    # Metrics counters
    tp = 0  # Defect correctly flagged DEFECT
    tn = 0  # Good correctly passed PASS
    fp = 0  # Good incorrectly flagged DEFECT
    fn = 0  # Defect incorrectly passed PASS
    total_time = 0.0

    cat_stats = {}
    current_cat = None

    for idx, item in enumerate(items, start=1):
        cat = item["category"]
        rel_path = item["filename"]
        gt = item["ground_truth"]
        defect_type = item["defect_type"]

        if cat not in cat_stats:
            cat_stats[cat] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0, "time_ms": 0.0, "good": 0, "defect": 0}
        cat_stats[cat]["total"] += 1
        if gt == "DEFECT":
            cat_stats[cat]["defect"] += 1
        else:
            cat_stats[cat]["good"] += 1

        img_path = stream_dir / cat / rel_path
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if cat != current_cat:
            engine.set_category(cat)
            current_cat = cat

        # Run inference (default full memory bank for maximum accuracy; --fast for quick preview)
        t_iter = time.perf_counter()
        result = engine.inspect(img, fast_mode=args.fast)
        dur_ms = (time.perf_counter() - t_iter) * 1000.0
        total_time += dur_ms
        cat_stats[cat]["time_ms"] += dur_ms

        score = result["anomaly_score"]
        tau = result["threshold"]

        # Pure Binary Decision: PASS vs REJECT
        is_reject = (score >= tau)
        verdict_str = "REJECT" if is_reject else "PASS"

        # Evaluate against Ground Truth
        if gt == "DEFECT":
            if is_reject:
                tp += 1
                cat_stats[cat]["tp"] += 1
                eval_res = "CORRECT (TP)"
                res_str = "\033[92mMATCH [TP]\033[0m"
            else:
                fn += 1
                cat_stats[cat]["fn"] += 1
                eval_res = "ESCAPE (FN)"
                res_str = "\033[91mESCAPE [FN]\033[0m"
        else: # GOOD
            if not is_reject:
                tn += 1
                cat_stats[cat]["tn"] += 1
                eval_res = "CORRECT (TN)"
                res_str = "\033[92mMATCH [TN]\033[0m"
            else:
                fp += 1
                cat_stats[cat]["fp"] += 1
                eval_res = "FALSE_ALARM (FP)"
                res_str = "\033[91mFALSE [FP]\033[0m"

        # Color verdict string
        v_str = "\033[91mREJECT   \033[0m" if is_reject else "\033[92mPASS [OK]\033[0m"

        short_file = rel_path if len(rel_path) <= 16 else rel_path[-16:]

        print(
            f"#{idx:<5} | {cat:<14} | {short_file:<16} | {gt:<7} | {score:6.3f}  | {tau:5.2f} | "
            f"{v_str:<10} | {res_str:<21} | {dur_ms:6.1f} ms"
        )

        csv_writer.writerow([
            idx, cat, rel_path, gt, defect_type,
            round(score, 4), round(tau, 4), verdict_str, eval_res, round(dur_ms, 2)
        ])
        csv_file.flush()

        if args.delay > 0:
            time.sleep(args.delay)

    csv_file.close()

    # 4. Final Industrial Audit Summary
    total = len(items)
    accuracy = ((tp + tn) / max(total, 1)) * 100.0
    precision = (tp / max(tp + fp, 1)) * 100.0
    recall = (tp / max(tp + fn, 1)) * 100.0

    print("\n" + "=" * 115)
    print(" AUTOVI INDUSTRIAL EVALUATION SUMMARY AUDIT (HAILO-8 NPU)")
    print("=" * 115)
    print(f"{'CATEGORY':<18} | {'TOTAL':<6} | {'GOOD':<5} | {'DEFECT':<6} | {'TP':<4} | {'TN':<4} | {'FP':<4} | {'FN':<4} | {'ACCURACY':<9} | {'RECALL':<8} | {'AVG LAT':<8}")
    print("-" * 115)

    summary_rows = []
    for c_name, s in cat_stats.items():
        c_tot = s["total"]
        c_acc = ((s["tp"] + s["tn"]) / max(c_tot, 1)) * 100.0
        c_rec = (s["tp"] / max(s["tp"] + s["fn"], 1)) * 100.0
        c_lat = s["time_ms"] / max(c_tot, 1)
        print(f"{c_name:<18} | {c_tot:<6} | {s['good']:<5} | {s['defect']:<6} | {s['tp']:<4} | {s['tn']:<4} | {s['fp']:<4} | {s['fn']:<4} | {c_acc:8.2f}% | {c_rec:7.2f}% | {c_lat:6.1f} ms")
        summary_rows.append([
            c_name, c_tot, s["good"], s["defect"], s["tp"], s["tn"], s["fp"], s["fn"],
            round(c_acc, 2), round(c_rec, 2), round(c_lat, 2)
        ])

    print("-" * 115)
    print(f"{'OVERALL':<18} | {total:<6} | {sum(s['good'] for s in cat_stats.values()):<5} | {sum(s['defect'] for s in cat_stats.values()):<6} | {tp:<4} | {tn:<4} | {fp:<4} | {fn:<4} | {accuracy:8.2f}% | {recall:7.2f}% | {total_time / max(total, 1):6.1f} ms")
    print("=" * 115)
    print(f" Total Images Evaluated:           {total}")
    print(f" True Positives (Defects Caught):  {tp} ({recall:.1f}%)")
    print(f" True Negatives (Good Parts Pass): {tn} ({tn / max(sum(s['good'] for s in cat_stats.values()), 1) * 100:.1f}%)")
    print(f" False Alarms (False Positives):   {fp}")
    print(f" Defect Escapes (False Negatives): {fn}")
    print(f" Overall Inspection Accuracy:      {accuracy:.2f}%")
    print(f" Average Latency per Frame:        {total_time / max(total, 1):.1f} ms")
    print(f" Detailed Audit CSV Saved To:      {csv_path.resolve()}")

    # Save summary CSV
    sum_path = Path(args.summary_csv)
    with open(sum_path, "w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow(["category", "total_images", "good", "defect", "true_positives", "true_negatives", "false_positives", "false_negatives", "accuracy_pct", "recall_pct", "avg_latency_ms"])
        for sr in summary_rows:
            sw.writerow(sr)
        sw.writerow(["OVERALL", total, sum(s['good'] for s in cat_stats.values()), sum(s['defect'] for s in cat_stats.values()), tp, tn, fp, fn, round(accuracy, 2), round(recall, 2), round(total_time / max(total, 1), 2)])
    print(f" Summary Metric CSV Saved To:      {sum_path.resolve()}")
    print("=" * 115)


if __name__ == "__main__":
    main()
