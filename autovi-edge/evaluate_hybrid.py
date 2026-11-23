#!/usr/bin/env python3
"""AutoVI Edge: Hybrid Benchmark (Hailo-8 NPU for 3 classes + ARM CPU for 3 classes).

Hardware Mapping:
- Hailo-8 NPU (26 TOPS): tank_screw, underbody_pipes, underbody_screw
- ARM Cortex-A76 (ONNX): pipe_clip, pipe_staple, engine_wiring
"""

import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
import cv2

EDGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EDGE_ROOT))

from src.inference_engine import EdgeAnomalyEngine

HAILO_CATEGORIES = ["tank_screw", "underbody_pipes", "underbody_screw"]
CPU_CATEGORIES   = ["pipe_clip", "pipe_staple", "engine_wiring"]

def main():
    parser = argparse.ArgumentParser(description="AutoVI Hybrid Evaluator")
    parser.add_argument("--limit", type=int, default=0, help="Limit images per category (0 = all)")
    parser.add_argument("--csv", default="hybrid_results.csv", help="Detailed CSV output")
    parser.add_argument("--summary-csv", default="hybrid_summary.csv", help="Summary CSV output")
    args = parser.parse_args()

    stream_dir = EDGE_ROOT / "test_stream"
    gt_file = stream_dir / "ground_truth.csv"
    if not gt_file.exists():
        sys.exit(f"[ERROR] Ground truth manifest not found: {gt_file}")

    print("=" * 115)
    print(" AUTOVI EDGE: HYBRID 6-CATEGORY BENCHMARK")
    print(" Hailo-8 NPU (26 TOPS): tank_screw, underbody_pipes, underbody_screw")
    print(" ARM Cortex-A76 (ONNX): pipe_clip, pipe_staple, engine_wiring")
    print("=" * 115)

    # Load ground truth items
    all_items = []
    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_items.append(row)

    # Initialize both engines
    print("\n[1/2] Initializing Hailo-8 NPU Backend...")
    try:
        hailo_engine = EdgeAnomalyEngine(model_dir=EDGE_ROOT / "models", preferred_backend="hailo")
        print("      Hailo-8 NPU Ready.")
    except Exception as e:
        print(f"      [WARNING] Hailo initialization error: {e}. Falling back to ONNX.")
        hailo_engine = EdgeAnomalyEngine(model_dir=EDGE_ROOT / "models", preferred_backend="onnx")

    print("\n[2/2] Initializing ARM Cortex-A76 ONNX Backend...")
    onnx_engine = EdgeAnomalyEngine(model_dir=EDGE_ROOT / "models", preferred_backend="onnx")
    print("      ARM ONNX Backend Ready.\n")

    cat_stats = {
        cat: {
            "total": 0, "good": 0, "defect": 0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "time_ms": 0.0,
            "device": "Hailo-8" if cat in HAILO_CATEGORIES else "ARM CPU"
        }
        for cat in HAILO_CATEGORIES + CPU_CATEGORIES
    }

    # Group items by category
    from collections import defaultdict
    cat_items = defaultdict(list)
    for row in all_items:
        cat_items[row["category"]].append(row)

    total_evaluated = 0
    total_time = 0.0

    csv_out = open(args.csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["category", "target_device", "filename", "ground_truth", "defect_type", "score", "threshold", "verdict", "result", "latency_ms"])

    for cat in CPU_CATEGORIES + HAILO_CATEGORIES:
        items = cat_items.get(cat, [])
        if args.limit > 0:
            items = items[:args.limit]
        if not items:
            continue

        target_dev = "Hailo-8" if cat in HAILO_CATEGORIES else "ARM CPU"
        engine = hailo_engine if cat in HAILO_CATEGORIES else onnx_engine
        engine.set_category(cat)
        tau = engine.category_models[cat]["threshold"]

        print(f"\n--- Category: {cat.upper()} ({len(items)} images) -> Running on {target_dev} (tau={tau:.2f}) ---")

        for idx, row in enumerate(items, 1):
            rel_path = row["filename"]
            gt = row["ground_truth"].upper()
            defect_type = row.get("defect_type", "nominal")
            img_path = stream_dir / cat / rel_path

            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            t0 = time.perf_counter()
            res = engine.inspect(img, fast_mode=False)
            dur_ms = (time.perf_counter() - t0) * 1000.0

            score = res["anomaly_score"]
            is_reject = (score >= tau)
            verdict_str = "REJECT" if is_reject else "PASS"

            # Update stats
            s = cat_stats[cat]
            s["total"] += 1
            s["time_ms"] += dur_ms
            total_time += dur_ms
            total_evaluated += 1

            if gt == "DEFECT":
                s["defect"] += 1
                if is_reject:
                    s["tp"] += 1
                    eval_res = "CORRECT (TP)"
                else:
                    s["fn"] += 1
                    eval_res = "ESCAPE (FN)"
            else:
                s["good"] += 1
                if not is_reject:
                    s["tn"] += 1
                    eval_res = "CORRECT (TN)"
                else:
                    s["fp"] += 1
                    eval_res = "FALSE_ALARM (FP)"

            if idx % 50 == 0 or idx == len(items):
                acc = ((s["tp"] + s["tn"]) / max(s["total"], 1)) * 100.0
                rec = (s["tp"] / max(s["defect"], 1)) * 100.0 if s["defect"] > 0 else 100.0
                print(f"  [{idx:4d}/{len(items):4d}] Progress: {acc:5.1f}% Acc | {dur_ms:5.1f} ms/frame")

            csv_writer.writerow([cat, target_dev, rel_path, gt, defect_type, round(score, 4), round(tau, 4), verdict_str, eval_res, round(dur_ms, 2)])

    csv_out.close()

    # Final summary audit table
    print("\n" + "=" * 125)
    print(" AUTOVI INDUSTRIAL HYBRID EVALUATION AUDIT SUMMARY")
    print("=" * 125)
    print(f"{'CATEGORY':<18} | {'TARGET':<9} | {'TOTAL':<6} | {'GOOD':<5} | {'DEFECT':<6} | {'TP':<4} | {'TN':<4} | {'FP':<4} | {'FN':<4} | {'ACCURACY':<9} | {'RECALL':<8} | {'AVG LAT':<8}")
    print("-" * 125)

    tot_tp = sum(s["tp"] for s in cat_stats.values())
    tot_tn = sum(s["tn"] for s in cat_stats.values())
    tot_fp = sum(s["fp"] for s in cat_stats.values())
    tot_fn = sum(s["fn"] for s in cat_stats.values())
    tot_good = sum(s["good"] for s in cat_stats.values())
    tot_def = sum(s["defect"] for s in cat_stats.values())
    ov_acc = ((tot_tp + tot_tn) / max(total_evaluated, 1)) * 100.0
    ov_rec = (tot_tp / max(tot_def, 1)) * 100.0 if tot_def > 0 else 0.0

    summary_rows = []
    for c_name, s in cat_stats.items():
        c_tot = s["total"]
        if c_tot == 0:
            continue
        c_acc = ((s["tp"] + s["tn"]) / max(c_tot, 1)) * 100.0
        c_rec = (s["tp"] / max(s["defect"], 1)) * 100.0 if s["defect"] > 0 else 100.0
        c_lat = s["time_ms"] / max(c_tot, 1)
        print(f"{c_name:<18} | {s['device']:<9} | {c_tot:<6} | {s['good']:<5} | {s['defect']:<6} | {s['tp']:<4} | {s['tn']:<4} | {s['fp']:<4} | {s['fn']:<4} | {c_acc:8.2f}% | {c_rec:7.2f}% | {c_lat:6.1f} ms")
        summary_rows.append([c_name, s["device"], c_tot, s["good"], s["defect"], s["tp"], s["tn"], s["fp"], s["fn"], round(c_acc, 2), round(c_rec, 2), round(c_lat, 2)])

    print("-" * 125)
    print(f"{'OVERALL':<18} | {'HYBRID':<9} | {total_evaluated:<6} | {tot_good:<5} | {tot_def:<6} | {tot_tp:<4} | {tot_tn:<4} | {tot_fp:<4} | {tot_fn:<4} | {ov_acc:8.2f}% | {ov_rec:7.2f}% | {total_time / max(total_evaluated, 1):6.1f} ms")
    print("=" * 125)
    print(f" Saved Detailed CSV: {Path(args.csv).resolve()}")

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow(["category", "target", "total", "good", "defect", "tp", "tn", "fp", "fn", "accuracy_pct", "recall_pct", "avg_lat_ms"])
        for r in summary_rows:
            sw.writerow(r)
        sw.writerow(["OVERALL", "HYBRID", total_evaluated, tot_good, tot_def, tot_tp, tot_tn, tot_fp, tot_fn, round(ov_acc, 2), round(ov_rec, 2), round(total_time / max(total_evaluated, 1), 2)])

if __name__ == "__main__":
    main()
