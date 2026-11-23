#!/usr/bin/env python3
"""AutoVI Edge: Continuous Live Anomaly Detection Benchmark.
Continuously runs PatchCore inference with real-time FPS, latency, and verdict telemetry.
Supports real dataset test images (good / defect) and synthetic frames.
"""

import argparse
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
    parser = argparse.ArgumentParser(description="AutoVI Continuous Live Benchmark")
    parser.add_argument("--category", default="tank_screw", choices=[
        "tank_screw", "pipe_clip", "engine_wiring", "pipe_staple", "underbody_pipes", "underbody_screw"
    ])
    parser.add_argument("--backend", default="auto", choices=["auto", "hailo", "onnx"])
    parser.add_argument("--device", default="auto", help="Alias for backend")
    parser.add_argument("--mode", default="cycle", choices=["cycle", "good", "defect", "noise"],
                        help="'good': test good parts only, 'defect': test defective parts only, 'cycle': alternate between both, 'noise': random synthetic")
    parser.add_argument("--binary", action="store_true",
                        help="Use binary 2-tier classification (PASS / DEFECT) without the 15% marginal review buffer")
    args = parser.parse_args()

    backend = args.backend if args.backend != "auto" else args.device

    print("=" * 88)
    print(" AUTOVI EDGE: CONTINUOUS LIVE INFERENCE BENCHMARK")
    print(f" Category: {args.category} | Backend: {backend} | Input Mode: {args.mode.upper()}")
    print("=" * 88)

    model_dir = EDGE_ROOT / "models"
    config_dir = EDGE_ROOT / "configs"

    print("Initializing inference engine...")
    t0 = time.perf_counter()
    engine = EdgeAnomalyEngine(
        model_dir=model_dir,
        preferred_backend=backend,
    )
    engine.set_category(args.category)
    init_time = time.perf_counter() - t0
    print(f"Engine Ready in {init_time:.2f}s | Active Backend: {engine.backend_name}\n")

    decision_engine = DecisionEngine()

    # Load real sample images
    samples_dir = EDGE_ROOT / "sample_images"
    good_files = sorted(list((samples_dir / "good").glob("*.png"))) if (samples_dir / "good").exists() else []
    defect_files = sorted(list((samples_dir / "defect").glob("*.png"))) if (samples_dir / "defect").exists() else []

    loaded_good = [(f.name, cv2.imread(str(f))) for f in good_files if cv2.imread(str(f)) is not None]
    loaded_defect = [(f.name, cv2.imread(str(f))) for f in defect_files if cv2.imread(str(f)) is not None]

    print(f"Sample images loaded: {len(loaded_good)} nominal GOOD images, {len(loaded_defect)} DEFECT images.")

    # Synthetic fallback if no images found
    np.random.seed(42)
    noise_frame = ("noise", np.random.randint(40, 215, (480, 640, 3), dtype=np.uint8))

    print(f"{'FRAME':<7} | {'FILE':<10} | {'TYPE':<7} | {'BACKEND':<10} | {'FPS':<5} | {'LATENCY':<9} | {'BACKBONE':<9} | {'NN SEARCH':<9} | {'SCORE':<7} | {'VERDICT':<10}")
    print("-" * 110)

    frame_idx = 0
    t_start = time.perf_counter()
    fps_ema = 0.0

    try:
        while True:
            frame_idx += 1
            t_iter = time.perf_counter()

            # Select frame based on mode
            if args.mode == "good":
                img_type = "GOOD"
                img_name, frame = loaded_good[frame_idx % len(loaded_good)] if loaded_good else noise_frame
            elif args.mode == "defect":
                img_type = "DEFECT"
                img_name, frame = loaded_defect[frame_idx % len(loaded_defect)] if loaded_defect else noise_frame
            elif args.mode == "noise":
                img_type = "NOISE"
                img_name, frame = noise_frame
            else:  # cycle
                # Show 5 good images, then 5 defect images
                cycle_step = (frame_idx // 5) % 2
                if cycle_step == 0 and loaded_good:
                    img_type = "GOOD"
                    img_name, frame = loaded_good[frame_idx % len(loaded_good)]
                elif loaded_defect:
                    img_type = "DEFECT"
                    img_name, frame = loaded_defect[frame_idx % len(loaded_defect)]
                else:
                    img_type = "NOISE"
                    img_name, frame = noise_frame

            # Run inspection
            result = engine.inspect(frame)
            score = result["anomaly_score"]
            tau = result["threshold"]
            timings = result["timings_ms"]
            t_total_ms = timings.get("total", (time.perf_counter() - t_iter) * 1000.0)
            t_fe_ms = timings.get("feature_extraction", 0.0)
            t_nn_ms = timings.get("nearest_neighbor", 0.0)

            # Compute FPS
            iter_dur = time.perf_counter() - t_iter
            inst_fps = 1.0 / max(iter_dur, 1e-4)
            fps_ema = inst_fps if frame_idx == 1 else 0.9 * fps_ema + 0.1 * inst_fps

            # Get verdict
            verdict_info = decision_engine.evaluate(args.category, score)
            verdict = verdict_info.verdict

            # Pure Binary Verdict: PASS vs REJECT (No review cases)
            if score < tau:
                v_str = "\033[92mPASS [OK]\033[0m"
            else:
                v_str = "\033[91mREJECT   \033[0m"

            b_name = "Hailo NPU" if "hailo" in engine.backend_name.lower() else "ONNX CPU"

            print(
                f"#{frame_idx:<6} | {img_name:<10} | {img_type:<7} | {b_name:<10} | {fps_ema:4.1f} | {t_total_ms:6.1f} ms | "
                f"{t_fe_ms:6.1f} ms | {t_nn_ms:6.1f} ms | {score:6.4f}  | {v_str}"
            )

            # Small yield
            time.sleep(0.01)

    except KeyboardInterrupt:
        total_time = time.perf_counter() - t_start
        avg_fps = frame_idx / max(total_time, 1e-4)
        print("\n" + "=" * 97)
        print(f"Benchmark stopped by user.")
        print(f"Total Frames Processed: {frame_idx}")
        print(f"Elapsed Time:           {total_time:.2f} s")
        print(f"Average Throughput:     {avg_fps:.2f} FPS")
        print("=" * 97)


if __name__ == "__main__":
    main()
