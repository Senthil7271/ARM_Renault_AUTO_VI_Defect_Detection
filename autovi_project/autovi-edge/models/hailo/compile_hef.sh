#!/bin/bash
# ==============================================================================
# AutoVI: Reproducible Hailo-8 HEF Compilation Script
# Target Accelerator: Hailo-8 (26 TOPS, 8-cluster architecture)
# Host Architecture: Linux x86_64
# Compatible HailoRT: v4.17.0 / v4.18.0 (Raspberry Pi AI HAT+)
# ==============================================================================

set -e

echo "================================================================================"
echo "HAILO DATAFLOW COMPILER: WIDERESNET-50-2 PATCHCORE (HAILO-8 26 TOPS)"
echo "================================================================================"

ONNX_MODEL="/workspace/models/onnx/wideresnet50_patchcore.onnx"
ALLS_SCRIPT="/workspace/configs/wideresnet50_patchcore.alls"
CALIB_DATA="/workspace/models/hailo/calibration_dataset.npy"
OUTPUT_DIR="/workspace/models/hailo"

# Output artifact paths
PARSED_HAR="${OUTPUT_DIR}/wideresnet50_patchcore_parsed.har"
QUANT_HAR="${OUTPUT_DIR}/wideresnet50_patchcore_quantized.har"
FINAL_HEF="${OUTPUT_DIR}/wideresnet50_patchcore.hef"

# 1. Verification of inputs
if [ ! -f "$ONNX_MODEL" ]; then
    echo "[ERROR] Input ONNX model not found: $ONNX_MODEL"
    exit 1
fi

if [ ! -f "$ALLS_SCRIPT" ]; then
    echo "[ERROR] Hailo model script not found: $ALLS_SCRIPT"
    exit 1
fi

if [ ! -f "$CALIB_DATA" ]; then
    echo "[ERROR] Calibration dataset not found: $CALIB_DATA"
    exit 1
fi

echo "[INPUTS VERIFIED]"
echo "  ONNX:   $ONNX_MODEL"
echo "  ALLS:   $ALLS_SCRIPT"
echo "  CALIB:  $CALIB_DATA"

# ------------------------------------------------------------------------------
# STEP 1: Parse ONNX graph into Hailo Archive (HAR)
# Target: hailo8 (NOT hailo8l)
# ------------------------------------------------------------------------------
echo ""
echo "[1/3] Parsing ONNX Model to Hailo Archive (HAR)..."
hailo parser onnx "$ONNX_MODEL" \
    --hw-arch hailo8 \
    --output-har-path "$PARSED_HAR"

# ------------------------------------------------------------------------------
# STEP 2: Optimize and Quantize (Post-Training Quantization)
# Applies ImageNet normalization in-hardware and calibrates activation ranges
# using representative normal training samples across all 6 AutoVI categories
# ------------------------------------------------------------------------------
echo ""
echo "[2/3] Optimizing & Quantizing using Normal Calibration Dataset (PTQ)..."
hailo optimize "$PARSED_HAR" \
    --hw-arch hailo8 \
    --calib-set-path "$CALIB_DATA" \
    --model-script "$ALLS_SCRIPT" \
    --output-har-path "$QUANT_HAR"

# ------------------------------------------------------------------------------
# STEP 3: Compile into Hailo Executable Format (HEF)
# Allocates physical Hailo-8 compute clusters and memory blocks
# Exposes output_patch_embeddings [1, 32, 32, 1536]
# ------------------------------------------------------------------------------
echo ""
echo "[3/3] Compiling Quantized Network to HEF (Hailo-8 26 TOPS)..."
hailo compiler "$QUANT_HAR" \
    --hw-arch hailo8 \
    --output-hef-path "$FINAL_HEF"

echo ""
echo "================================================================================"
echo "[COMPILATION SUCCESSFUL]"
echo "Compiled HEF artifact: $FINAL_HEF"
ls -lh "$FINAL_HEF"
echo "================================================================================"
