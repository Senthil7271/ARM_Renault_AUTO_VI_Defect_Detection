#!/bin/bash
# ==============================================================================
# AutoVI Edge: Automated Setup Script for Raspberry Pi 5 + Hailo-8 AI HAT+
# Target OS: Raspberry Pi OS 64-bit (Debian Bookworm)
# ==============================================================================

set -e

# Parse command line options
OPT_ENABLE_GEN3=false
OPT_INSTALL_SERVICE=false
OPT_ASSUME_YES=false

for arg in "$@"; do
    case "$arg" in
        --enable-pcie-gen3)
            OPT_ENABLE_GEN3=true
            ;;
        --enable-service)
            OPT_INSTALL_SERVICE=true
            ;;
        -y|--yes)
            OPT_ASSUME_YES=true
            ;;
        --help|-h)
            echo "Usage: ./install.sh [OPTIONS]"
            echo "Options:"
            echo "  --enable-pcie-gen3   Configure Raspberry Pi 5 PCIe Gen 3 (8.0 GT/s) in config.txt"
            echo "  --enable-service     Install systemd background service (kept disabled by default)"
            echo "  -y, --yes            Non-interactive mode (defaults to NO for optional system changes)"
            exit 0
            ;;
    esac
done

echo "================================================================================"
echo "INSTALLING AUTOVI EDGE ON RASPBERRY PI 5 (AI HAT+ READY)"
echo "================================================================================"

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/autovi_venv"

# ------------------------------------------------------------------------------
# 1. Update system packages & Install Hardware Drivers
# (Requires Internet access on first-time setup to fetch APT packages)
# ------------------------------------------------------------------------------
echo ""
echo "[1/5] Installing Raspberry Pi OS system dependencies & Hailo PCIe stack..."
echo "  Note: Requires active internet connection to contact official Raspberry Pi APT mirrors."
sudo apt-get update
sudo apt-get install -y \
    hailo-all \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libatlas-base-dev \
    v4l-utils \
    pciutils

# ------------------------------------------------------------------------------
# 2. Configure Raspberry Pi 5 PCIe Gen 3 Speed (OPTIONAL)
# Default: PCIe Gen 2.0 (5.0 GT/s) is preserved unless user opts in
# ------------------------------------------------------------------------------
echo ""
echo "[2/5] Checking PCIe Bus Configuration..."
CONFIG_TXT="/boot/firmware/config.txt"

SHOULD_CONFIG_GEN3=false
if [ "$OPT_ENABLE_GEN3" = true ]; then
    SHOULD_CONFIG_GEN3=true
elif [ "$OPT_ASSUME_YES" = false ] && [ -t 0 ]; then
    echo "  Notice: Raspberry Pi 5 defaults to PCIe Gen 2.0 (5.0 GT/s)."
    echo "          PCIe Gen 3.0 (8.0 GT/s) unlocks maximum bandwidth for 26 TOPS."
    read -p "  Do you want to configure PCIe Gen 3 in /boot/firmware/config.txt? [y/N]: " REPLY_GEN3
    if [[ "$REPLY_GEN3" =~ ^[Yy]$ ]]; then
        SHOULD_CONFIG_GEN3=true
    fi
fi

if [ "$SHOULD_CONFIG_GEN3" = true ] && [ -f "$CONFIG_TXT" ]; then
    if ! grep -q "dtparam=pciex1" "$CONFIG_TXT"; then
        echo "  Configuring dtparam=pciex1 and dtparam=pciex1_gen=3 in $CONFIG_TXT..."
        echo "" | sudo tee -a "$CONFIG_TXT"
        echo "# AutoVI: Hailo-8 AI HAT+ PCIe Gen 3 Configuration" | sudo tee -a "$CONFIG_TXT"
        echo "dtparam=pciex1" | sudo tee -a "$CONFIG_TXT"
        echo "dtparam=pciex1_gen=3" | sudo tee -a "$CONFIG_TXT"
        echo "  [OK] PCIe Gen 3 configured. (Requires reboot to take effect)"
    else
        echo "  [INFO] PCIe configuration already present in $CONFIG_TXT."
    fi
else
    echo "  [SKIPPED] PCIe Gen 3 configuration skipped. Running at standard PCIe Gen 2 speed."
fi

# ------------------------------------------------------------------------------
# 3. Create clean Python Virtual Environment
# ------------------------------------------------------------------------------
echo ""
echo "[3/5] Setting up offline Python virtual environment ($VENV_DIR)..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------------------------
# 4. Install Python dependencies
# (Requires Internet access on first-time setup to fetch PyPI packages)
# ------------------------------------------------------------------------------
echo ""
echo "[4/5] Installing Python runtime dependencies..."
echo "  Note: Requires internet access to download wheels from PyPI."
pip install --upgrade pip
pip install \
    "numpy>=1.23.0" \
    "opencv-python-headless>=4.8.0" \
    "scipy>=1.10.0" \
    "onnxruntime>=1.16.0" \
    "flask>=2.3.0"

# Make all shell scripts and tools executable
chmod +x "$CURRENT_DIR"/*.sh 2>/dev/null || true
if [ -d "$CURRENT_DIR/tools" ]; then
    chmod +x "$CURRENT_DIR"/tools/*.sh "$CURRENT_DIR"/tools/*.py 2>/dev/null || true
fi

# ------------------------------------------------------------------------------
# 5. Run Hardware Diagnostics and Commissioning Self-Test
# ------------------------------------------------------------------------------
echo ""
echo "[5/5] Running hardware diagnostics & commissioning self-test..."
if [ -f "$CURRENT_DIR/tools/hailo_diagnostics.sh" ]; then
    echo "--- Running Hailo Hardware Diagnostics ---"
    bash "$CURRENT_DIR/tools/hailo_diagnostics.sh" || true
fi

echo ""
echo "--- Running AutoVI System Self-Test ---"
SELF_TEST_PASSED=false
if python "$CURRENT_DIR/self_test.py"; then
    SELF_TEST_PASSED=true
    echo "  [PASS] Commissioning self-test completed successfully!"
else
    echo "  [WARNING] Self-test reported issues. Review error output above."
fi

# ------------------------------------------------------------------------------
# Optional: Systemd Service Setup (DISABLED by default)
# Only offered/installed if commissioning self-test passed
# ------------------------------------------------------------------------------
echo ""
echo "--------------------------------------------------------------------------------"
echo "SYSTEMD SERVICE AUTO-START CONFIGURATION"
echo "--------------------------------------------------------------------------------"
SERVICE_FILE="/etc/systemd/system/autovi-edge.service"

SHOULD_INSTALL_SERVICE=false
if [ "$SELF_TEST_PASSED" = true ]; then
    if [ "$OPT_INSTALL_SERVICE" = true ]; then
        SHOULD_INSTALL_SERVICE=true
    elif [ "$OPT_ASSUME_YES" = false ] && [ -t 0 ]; then
        read -p "Commissioning passed. Install auto-start systemd service (kept DISABLED by default)? [y/N]: " REPLY_SVC
        if [[ "$REPLY_SVC" =~ ^[Yy]$ ]]; then
            SHOULD_INSTALL_SERVICE=true
        fi
    fi
else
    echo "  [NOTICE] Systemd service installation blocked because self-test did not pass 100%."
fi

if [ "$SHOULD_INSTALL_SERVICE" = true ]; then
    echo "Creating systemd unit: $SERVICE_FILE..."
    sudo bash -c "cat <<EOF > $SERVICE_FILE
[Unit]
Description=AutoVI Edge Inspection Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$CURRENT_DIR
ExecStart=$CURRENT_DIR/start.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"
    sudo systemctl daemon-reload
    # Ensure it remains DISABLED until user explicitly enables it
    sudo systemctl disable autovi-edge.service 2>/dev/null || true
    echo "  [OK] Service installed at $SERVICE_FILE (DISABLED)."
    echo "  To enable auto-start on boot, run: sudo systemctl enable --now autovi-edge.service"
else
    echo "  [SKIPPED] Systemd service not installed."
fi

echo ""
echo "================================================================================"
echo "[SUCCESS] AutoVI Edge Installation Complete!"
echo "To start the live dashboard manually, run:"
echo "  ./start.sh"
echo "================================================================================"
