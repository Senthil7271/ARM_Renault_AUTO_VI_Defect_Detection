"""Inference latency and throughput benchmarking script.

Usage:
    python scripts/benchmark.py --category pipe_clip [--runs 50]
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchcore_model import PatchCoreModel


def main():
    parser = argparse.ArgumentParser(description="Benchmark PatchCore inference latency")
    parser.add_argument(
        "--category",
        type=str,
        default="pipe_clip",
        help="Category name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model .pkl file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of inference benchmark iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmark",
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else (
        PROJECT_ROOT / "outputs" / "experiments" / args.category / "model" / "patchcore_model.pkl"
    )

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Train the model first.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = PatchCoreModel.load(model_path)

    # Synthetic input batch of shape (B, 3, 256, 256)
    dummy_input = torch.randn(args.batch_size, 3, 256, 256, device=model.device)

    print(f"Warming up ({min(5, args.runs)} runs)...")
    for _ in range(min(5, args.runs)):
        _ = model.predict_batch(dummy_input)

    print(f"Benchmarking over {args.runs} runs (Batch Size: {args.batch_size})...")
    latencies = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _ = model.predict_batch(dummy_input)
        lat = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(lat)

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = (1000.0 / mean_lat) * args.batch_size

    print("\n============== Benchmark Results ==============")
    print(f"Device:           {model.device}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Mean Latency:     {mean_lat:.2f} ms (+/- {std_lat:.2f} ms)")
    print(f"Median Latency:   {np.median(latencies):.2f} ms")
    print(f"P95 Latency:      {np.percentile(latencies, 95):.2f} ms")
    print(f"Throughput (FPS): {fps:.2f} frames/second")
    print("================================================")


if __name__ == "__main__":
    main()
