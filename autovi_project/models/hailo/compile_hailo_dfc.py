#!/usr/bin/env python3
"""AutoVI: Automated Hailo-8 Compilation Pipeline via Hailo Dataflow Compiler (DFC) Python API.

Target Accelerator: Hailo-8 (26 TOPS, 8 clusters) for Raspberry Pi AI HAT+
Compatible HailoRT: v4.17.0 / v4.18.0
Host Architecture: Linux x86_64
Input Model: WideResNet-50-2 PatchCore Feature Extractor (ONNX)
Input Shape: [1, 3, 256, 256] float32
Target Shape: [1, 256, 256, 3] uint8 (with in-hardware ImageNet normalization)
Output Shape: [1, 32, 32, 1536] (concatenation of Layer2 512-dim + Layer3 1024-dim)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Search paths for inputs
DEFAULT_ONNX = PROJECT_ROOT / "models" / "onnx" / "wideresnet50_patchcore.onnx"
DEFAULT_ALLS = SCRIPT_DIR / "wideresnet50_patchcore.alls"
DEFAULT_CALIB = SCRIPT_DIR / "calibration_dataset.npy"
DEFAULT_OUT_HEF = SCRIPT_DIR / "wideresnet50_patchcore.hef"


def compile_patchcore_to_hailo8(
    onnx_path: Path = DEFAULT_ONNX,
    alls_path: Path = DEFAULT_ALLS,
    calib_path: Path = DEFAULT_CALIB,
    output_hef_path: Path = DEFAULT_OUT_HEF,
    hw_arch: str = "hailo8",
):
    print("=" * 80)
    print(f"HAILO DATAFLOW COMPILER: WideResNet-50-2 PatchCore -> {hw_arch.upper()} HEF")
    print("=" * 80)

    # 1. Prerequisite checks
    print("\n[STEP 1/4] Validating Compilation Inputs...")
    if not onnx_path.exists():
        sys.exit(f"[ERROR] Source ONNX model not found: {onnx_path}")
    if not alls_path.exists():
        sys.exit(f"[ERROR] Hailo model script (.alls) not found: {alls_path}")
    if not calib_path.exists():
        sys.exit(f"[ERROR] Normal calibration dataset (.npy) not found: {calib_path}")

    print(f"  Source ONNX:        {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  Model Script (.alls):{alls_path}")
    print(f"  Calibration Array:  {calib_path} ({calib_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  Target Architecture: {hw_arch}")

    # Check for Hailo SDK
    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        print("\n[CRITICAL ERROR] Hailo Dataflow Compiler (DFC) Python package not installed.")
        print("Hailo DFC is distributed exclusively as an x86_64 Linux wheel from:")
        print("  https://hailo.ai/developer-zone/software-downloads/")
        print("Package: hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl")
        print("\nTo run compilation:")
        print("  1. Build docker image: docker build -t hailo-dfc -f Dockerfile.hailo_dfc .")
        print("  2. Run compilation:    docker run --rm -v $(pwd)/../..:/workspace hailo-dfc")
        sys.exit(1)

    # 2. Parse ONNX Model
    print(f"\n[STEP 2/4] Parsing ONNX Compute Graph for {hw_arch}...")
    t0 = time.time()
    runner = ClientRunner(hw_arch=hw_arch)
    model_name = "wideresnet50_patchcore"
    
    runner.translate_onnx_model(
        str(onnx_path),
        model_name,
        start_node_names=["input"],
        end_node_names=["patch_features"],
        net_input_shapes={"input": [1, 3, 256, 256]},
    )
    parsed_har = SCRIPT_DIR / f"{model_name}_parsed.har"
    runner.save_har(str(parsed_har))
    print(f"  [PASS] Parsed graph saved: {parsed_har} ({time.time() - t0:.1f}s)")

    # 3. Optimize and Quantize (Post-Training Quantization)
    print("\n[STEP 3/4] Optimizing and Quantizing with Normal Calibration Dataset (PTQ)...")
    t1 = time.time()
    runner.load_model_script(str(alls_path))
    calib_data = np.load(calib_path)
    print(f"  Loaded calibration array: shape {calib_data.shape}, dtype {calib_data.dtype}")
    
    # Run optimization pass
    runner.optimize(calib_data)
    quant_har = SCRIPT_DIR / f"{model_name}_quantized.har"
    runner.save_har(str(quant_har))
    print(f"  [PASS] Quantized graph saved: {quant_har} ({time.time() - t1:.1f}s)")

    # 4. Compile to HEF binary
    print(f"\n[STEP 4/4] Allocating Physical Compute Units and Compiling to {hw_arch.upper()} HEF...")
    t2 = time.time()
    hef_bytes = runner.compile()
    
    with open(output_hef_path, "wb") as f:
        f.write(hef_bytes)

    out_size = output_hef_path.stat().st_size
    print(f"  [PASS] Compilation completed in {time.time() - t2:.1f}s")
    print(f"  HEF Binary: {output_hef_path}")
    print(f"  Size:       {out_size} bytes ({out_size / 1e6:.2f} MB)")

    # Sanity verification
    if out_size < 1_000_000:
        raise ValueError(f"Compiled HEF is unexpectedly small ({out_size} bytes)! Verify compiler logs.")

    print("\n" + "=" * 80)
    print("SUCCESS: Physical WideResNet-50-2 HEF generated and verified!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    compile_patchcore_to_hailo8()
