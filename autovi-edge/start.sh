#!/bin/bash
# ==============================================================================
# AutoVI Edge: Production Startup Script
# Raspberry Pi 5 + Hailo-8 AI HAT+ (26 TOPS)
# ==============================================================================

set -e

# Default Arguments
CATEGORY="${1:-tank_screw}"
BACKEND="${2:-auto}"
CAMERA="${3:-0}"
PORT="${4:-5000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate Virtual Environment if present
if [ -d "$HOME/autovi_venv" ]; then
    source "$HOME/autovi_venv/bin/activate"
fi

echo "================================================================================"
echo "LAUNCHING AUTOVI EDGE INSPECTION SERVER"
echo "================================================================================"
echo "  Category: $CATEGORY"
echo "  Backend:  $BACKEND"
echo "  Camera:   $CAMERA"
echo "  Web Port: $PORT"
echo "================================================================================"

python app.py \
    --category "$CATEGORY" \
    --backend "$BACKEND" \
    --camera "$CAMERA" \
    --port "$PORT" \
    --host "0.0.0.0"
