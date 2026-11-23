#!/usr/bin/env python3
"""AutoVI: First Real Inference Hardware Smoke Test for Hailo-8 on Raspberry Pi 5.

Executes physical inference on the Hailo-8 PCIe accelerator (26 TOPS):
1. Verifies physical HEF file integrity (rejects stubs).
2. Initializes VDevice and configures PCIe VStreams.
3. Performs a cold inference run.
4. Performs 10 consecutive warm inference runs and records hardware latency.
5. Verifies output embedding dimensions: (1, 32, 32, 1536) float32.
6. Asserts statistical validity of output features (mean, std, non-zero).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
EDGE_ROOT = TOOLS_DIR.parent
HEF_PATH = EDGE_ROOT / "models" / "wideresnet50_patchcore.hef"


def run_hailo_smoke_test():
    print("=" * 80)
    print("AUTOVI: HAILO-8 REAL HARDWARE INFERENCE SMOKE TEST (RASPBERRY PI 5)")
    print("=" * 80)

    # 1. Physical HEF check
    print("\n[STEP 1/5] Checking Physical HEF Artifact...")
    if not HEF_PATH.exists():
        sys.exit(f"[FAIL] HEF file not found: {HEF_PATH}")
    
    st_size = HEF_PATH.stat().st_size
    print(f"  HEF Path: {HEF_PATH}")
    print(f"  Size:     {st_size} bytes ({st_size / 1e6:.2f} MB)")
    if st_size < 1_000_000:
        sys.exit(
            f"[FAIL] REJECTED: File is an uncompiled stub ({st_size} bytes < 1 MB). "
            f"Physical HEF must be compiled with Hailo DFC on Linux x86_64."
        )
    print("  [PASS] Physical HEF artifact verified.")

    # 2. Check PyHailoRT availability
    print("\n[STEP 2/5] Initializing PyHailoRT Drivers...")
    try:
        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )
        print("  [PASS] Successfully imported 'hailo_platform'.")
    except ImportError as e:
        sys.exit(
            f"[FAIL] Failed to import 'hailo_platform': {e}\n"
            f"Ensure hailo-all is installed on Raspberry Pi OS: sudo apt install -y hailo-all"
        )

    # 3. Configure Hardware Device & VStreams
    print("\n[STEP 3/5] Configuring Hailo-8 PCIe Device & VStreams...")
    t_cfg = time.perf_counter()
    try:
        vdevice = VDevice()
        hef = HEF(str(HEF_PATH))
        cfg_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = vdevice.configure(hef, cfg_params)
        network_group = network_groups[0]
        net_params = network_group.create_params()

        input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        print(f"  [PASS] Configured in {time.perf_counter() - t_cfg:.3f}s")
        print(f"  Inputs:  {list(input_vstream_params.keys())}")
        print(f"  Outputs: {list(output_vstream_params.keys())}")
    except Exception as e:
        sys.exit(f"[FAIL] Hardware configuration failed: {e}")

    # 4. Synthesize input frame and run inference
    print("\n[STEP 4/5] Executing Real Inference on NPU Hardware...")
    input_name = list(input_vstream_params.keys())[0]
    output_name = list(output_vstream_params.keys())[0]

    # Deterministic test pattern
    np.random.seed(42)
    test_img = np.random.randint(0, 255, (1, 256, 256, 3), dtype=np.uint8)

    latencies = []
    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as pipeline:
        with network_group.activate(net_params):
            # Cold run
            t0 = time.perf_counter()
            out = pipeline.infer({input_name: test_img})
            cold_lat = (time.perf_counter() - t0) * 1000.0
            print(f"  [COLD RUN] Single-frame Latency: {cold_lat:.2f} ms")

            # 10 Warm runs
            for i in range(10):
                t_start = time.perf_counter()
                out = pipeline.infer({input_name: test_img})
                lat = (time.perf_counter() - t_start) * 1000.0
                latencies.append(lat)

    avg_lat = float(np.mean(latencies))
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
    print(f"  [WARM RUNS] 10-Iteration Mean Hardware Latency: {avg_lat:.2f} ms ({fps:.1f} FPS)")
    print(f"              Min: {min(latencies):.2f} ms, Max: {max(latencies):.2f} ms")

    # 5. Output tensor statistical verification
    print("\n[STEP 5/5] Verifying Output Feature Embeddings...")
    raw_emb = out[output_name]
    print(f"  Output Tensor Shape: {raw_emb.shape}, Dtype: {raw_emb.dtype}")

    # Expected shape: (1, 32, 32, 1536)
    if raw_emb.shape != (1, 32, 32, 1536):
        sys.exit(f"[FAIL] Unexpected output shape: {raw_emb.shape}, expected (1, 32, 32, 1536)")

    emb_mean = float(np.mean(raw_emb))
    emb_std = float(np.std(raw_emb))
    emb_min = float(np.min(raw_emb))
    emb_max = float(np.max(raw_emb))
    print(f"  Embedding Stats: mean={emb_mean:.4f}, std={emb_std:.4f}, min={emb_min:.4f}, max={emb_max:.4f}")

    if emb_std < 1e-4:
        sys.exit("[FAIL] Degenerate embedding: standard deviation is near zero (flat activations).")

    print("\n" + "=" * 80)
    print("SUCCESS: Hailo-8 Hardware Smoke Test PASSED! Real NPU acceleration confirmed.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    run_hailo_smoke_test()
