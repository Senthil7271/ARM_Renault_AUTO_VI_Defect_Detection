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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ONNX_MODEL="${PROJECT_ROOT}/models/onnx/wideresnet50_patchcore.onnx"
ALLS_SCRIPT="${SCRIPT_DIR}/wideresnet50_patchcore.alls"
CALIB_DATA="${SCRIPT_DIR}/calibration_dataset.npy"
OUTPUT_DIR="${SCRIPT_DIR}"

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
echo "  ONNX:   $ONNX_MODEL ($(du -h "$ONNX_MODEL" | cut -f1))"
echo "  ALLS:   $ALLS_SCRIPT"
echo "  CALIB:  $CALIB_DATA ($(du -h "$CALIB_DATA" | cut -f1))"

# Check if Python API compiler is preferred
if python3 -c "import hailo_sdk_client" >/dev/null 2>&1; then
    echo ""
    echo "[EXECUTING VIA PYTHON DFC API]"
    python3 "${SCRIPT_DIR}/compile_hailo_dfc.py"
    exit 0
fi

# Fallback to hailo CLI tool
if ! command -v hailo >/dev/null 2>&1; then
    echo "[ERROR] Hailo Dataflow Compiler ('hailo') CLI not found."
    echo "Hailo DFC requires an x86_64 Linux environment with the proprietary wheel installed."
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 1: Parse ONNX graph into Hailo Archive (HAR)
# Target: hailo8 (NOT hailo8l)
# ------------------------------------------------------------------------------
echo ""
echo "[1/3] Parsing ONNX Model to Hailo Archive (HAR)..."
hailo parser onnx "$ONNX_MODEL" \
    --hw-arch hailo8 \
    --start-node-names input \
    --end-node-names patch_features \
    --har-path "$PARSED_HAR" || \
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
    --output-har-path "$QUANT_HAR" || \
hailo optimize "$PARSED_HAR" \
    --hw-arch hailo8 \
    --calib-set-path "$CALIB_DATA" \
    --model-script "$ALLS_SCRIPT" \
    --har-path "$QUANT_HAR"

# ------------------------------------------------------------------------------
# STEP 3: Compile into Hailo Executable Format (HEF)
# Allocates physical Hailo-8 compute clusters and memory blocks
# Exposes patch_features [1, 32, 32, 1536]
# ------------------------------------------------------------------------------
echo ""
echo "[3/3] Compiling Quantized Network to HEF (Hailo-8 26 TOPS)..."
hailo compiler "$QUANT_HAR" \
    --hw-arch hailo8 \
    --output-hef-path "$FINAL_HEF" || \
hailo compiler "$QUANT_HAR" \
    --hw-arch hailo8 \
    --hef-path "$FINAL_HEF"

echo ""
echo "================================================================================"
echo "[COMPILATION SUCCESSFUL]"
echo "Compiled HEF artifact: $FINAL_HEF"
ls -lh "$FINAL_HEF"
echo "================================================================================"
