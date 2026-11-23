#!/usr/bin/env bash
# ==============================================================================
# AutoVI: Hailo-8 (26 TOPS) Hardware & Driver Diagnostic Verification Tool
# Target Hardware: Raspberry Pi 5 + Raspberry Pi AI HAT+ (26 TOPS, PCIe Gen 3)
# ==============================================================================

set -e

echo "=============================================================================="
echo "AUTOVI: HAILO-8 (26 TOPS) HARDWARE & DRIVER DIAGNOSTIC SUITE"
echo "Target: Raspberry Pi 5 + Raspberry Pi AI HAT+"
echo "=============================================================================="

ERRORS=0
WARNINGS=0

# ------------------------------------------------------------------------------
# 1. LAYER 1: PCIe Bus Hardware Link
# ------------------------------------------------------------------------------
echo ""
echo "[LAYER 1/4] Checking PCIe Hardware Link (lspci)..."
if ! command -v lspci &> /dev/null; then
    echo "  [WARNING] 'lspci' not found. Installing pciutils..."
    sudo apt update -qq && sudo apt install -y pciutils
fi

PCI_INFO=$(lspci -d 1e60: || true)
if [ -n "$PCI_INFO" ]; then
    echo "  [PASS] Hailo PCIe accelerator detected:"
    echo "         $PCI_INFO"
    
    # Check Link Speed & Width (Gen 2 vs Gen 3)
    LINK_STATUS=$(sudo lspci -vvv -d 1e60: 2>/dev/null | grep -E "LnkSta:" || true)
    if [ -n "$LINK_STATUS" ]; then
        echo "         $LINK_STATUS"
        if echo "$LINK_STATUS" | grep -qi "8.0GT/s"; then
            echo "  [OPTIMAL] PCIe Gen 3.0 (8.0 GT/s) link verified!"
        else
            echo "  [NOTICE] PCIe running at Gen 2.0 (5.0 GT/s). For 8.0 GT/s Gen 3:"
            echo "           Ensure 'dtparam=pciex1_gen=3' is in /boot/firmware/config.txt."
        fi
    fi
else
    echo "  [FAIL] Hailo PCIe device NOT detected on PCIe bus."
    echo "  Troubleshooting:"
    echo "  1. Check PCIe ribbon cable alignment on Pi 5 and AI HAT+."
    echo "  2. Confirm 'dtparam=pciex1' in /boot/firmware/config.txt."
    echo "  3. Reboot after modifying config.txt: sudo reboot"
    ERRORS=$((ERRORS + 1))
fi

# ------------------------------------------------------------------------------
# 2. LAYER 2: Kernel Driver & Device Node
# ------------------------------------------------------------------------------
echo ""
echo "[LAYER 2/4] Checking Hailo Kernel Driver & /dev/hailo0 Node..."
if [ -e "/dev/hailo0" ]; then
    echo "  [PASS] Device node '/dev/hailo0' exists."
    ls -l /dev/hailo0
else
    echo "  [FAIL] Device node '/dev/hailo0' does NOT exist."
    echo "  Kernel messages (dmesg):"
    dmesg | grep -i hailo | tail -n 5 || echo "  No Hailo kernel messages found."
    echo "  Run: sudo apt install -y hailo-all && sudo reboot"
    ERRORS=$((ERRORS + 1))
fi

# ------------------------------------------------------------------------------
# 3. LAYER 3: HailoRT Firmware & Architecture Identification
# ------------------------------------------------------------------------------
echo ""
echo "[LAYER 3/4] Querying Hailo Firmware & NPU Hardware Specs..."
if command -v hailortcli &> /dev/null; then
    HAILORT_VER=$(hailortcli --version 2>&1 || true)
    echo "  HailoRT CLI Version: $HAILORT_VER"
    
    FW_INFO=$(hailortcli fw-control identify 2>&1 || true)
    if echo "$FW_INFO" | grep -qi "Device Architecture.*HAILO8"; then
        echo "  [PASS] Firmware responded successfully:"
        echo "$FW_INFO" | sed 's/^/         /'
        if echo "$FW_INFO" | grep -qi "HAILO8L"; then
            echo "  [WARNING] Detected Hailo-8L (13 TOPS, 4 clusters). Expected full Hailo-8 (26 TOPS, 8 clusters)."
            WARNINGS=$((WARNINGS + 1))
        else
            echo "  [PASS] Confirmed: Full Hailo-8 (26 AI TOPS, 8 clusters) active!"
        fi
    else
        echo "  [FAIL] hailortcli failed to identify firmware: $FW_INFO"
        ERRORS=$((ERRORS + 1))
    fi

    # Measure PCIe transfer bandwidth
    echo ""
    echo "  Measuring PCIe DMA transfer bandwidth..."
    RATE_INFO=$(hailortcli pcie-rate-measure 2>&1 || true)
    if echo "$RATE_INFO" | grep -qi "Rate"; then
        echo "$RATE_INFO" | sed 's/^/         /'
    else
        echo "  [NOTICE] pcie-rate-measure returned: $RATE_INFO"
    fi
else
    echo "  [FAIL] 'hailortcli' command not found."
    echo "  Install via: sudo apt install -y hailo-all"
    ERRORS=$((ERRORS + 1))
fi

# ------------------------------------------------------------------------------
# 4. LAYER 4: HEF Artifact Verification
# ------------------------------------------------------------------------------
echo ""
echo "[LAYER 4/4] Validating Compiled HEF Artifact..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEF_PATH="${SCRIPT_DIR}/../models/wideresnet50_patchcore.hef"

if [ -f "$HEF_PATH" ]; then
    HEF_SIZE=$(wc -c < "$HEF_PATH")
    echo "  Found HEF: $HEF_PATH ($HEF_SIZE bytes)"
    if [ "$HEF_SIZE" -lt 1000000 ]; then
        echo "  [FAIL] HEF file is an uncompiled stub ($HEF_SIZE bytes < 1 MB)."
        echo "  Real WideResNet-50-2 HEF must be compiled with Hailo DFC on Linux x86_64."
        ERRORS=$((ERRORS + 1))
    else
        echo "  [PASS] HEF file size valid ($((HEF_SIZE / 1048576)) MB)."
        if command -v hailortcli &> /dev/null; then
            echo "  Parsing HEF architecture with hailortcli..."
            PARSE_OUT=$(hailortcli parse-hef "$HEF_PATH" 2>&1 || true)
            echo "$PARSE_OUT" | sed 's/^/         /'
            if echo "$PARSE_OUT" | grep -qi "HAILO8"; then
                echo "  [PASS] HEF verified for Hailo-8 architecture!"
            fi
        fi
    fi
else
    echo "  [NOTICE] Compiled HEF binary not found: $HEF_PATH"
    echo "           The application will operate via verified ONNX Runtime CPU fallback."
    echo "           To enable 26 TOPS Hailo-8 acceleration, compile wideresnet50_patchcore.onnx"
    echo "           using Hailo DFC on Linux x86_64 and place the resulting HEF at:"
    echo "           $HEF_PATH"
    WARNINGS=$((WARNINGS + 1))
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "=============================================================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "RESULT: ALL 4 LAYERS VERIFIED (100% OPERATIONAL HAILO-8 26 TOPS)"
elif [ $ERRORS -eq 0 ]; then
    echo "RESULT: HARDWARE OPERATIONAL WITH $WARNINGS WARNING(S)"
else
    echo "RESULT: DIAGNOSTIC FAILED WITH $ERRORS ERROR(S). Review logs above."
fi
echo "=============================================================================="
