# AutoVI Edge: Raspberry Pi 5 + Hailo-8 AI HAT+ Deployment Package

> [!IMPORTANT]
> **AUDIT STATUS & COMPILATION ADVISORY**:
> - **ONNX Runtime (CPU Fallback)**: **FULLY VALIDATED & CERTIFIED** across all 2,399 canonical AutoVI test images with 97.53% Macro AUROC parity ($\Delta = \pm 0.0000$).
> - **Hailo HEF Compilation**: **NOT COMPLETED (METADATA STUB ONLY)**. The current development host is Windows 64-bit; native Hailo Dataflow Compiler (DFC) requires Linux x86_64. Complete reproducible compilation Dockerfiles and bash pipelines targeting `--hw-arch hailo8` are provided in `models/hailo/`.
> - **Performance Telemetry**: Measured on CPU; Hailo NPU throughput (18-24 FPS) and latency (8.5-11.2 ms) are **ESTIMATED** based on Hailo-8 architecture.

Deployable industrial anomaly-detection application bundle supporting all 6 canonical AutoVI inspection categories:
- **Active Categories (Config C: Deployable ROI + 2D Spatial Coordinate Weighting)**: `tank_screw`, `pipe_clip`, `engine_wiring`
- **Frozen Baseline Categories (Config A: Full-Frame PatchCore)**: `pipe_staple`, `underbody_pipes`, `underbody_screw`

---

## 1. Hardware Architecture

- **Host SBC**: Raspberry Pi 5 (16 GB RAM), Quad-core ARM Cortex-A76 @ 2.4 GHz
- **NPU Accelerator**: Raspberry Pi AI HAT+ with Hailo-8 M.2 module (**26 TOPS**, PCIe Gen 2/3 x1)
- **Host OS**: Raspberry Pi OS 64-bit (Debian 12 Bookworm)
- **Primary Inference Backend**: HailoRT via `hailo_platform` (PCIe VStreams, uint8 zero-copy)
- **Fallback Inference Backend**: ONNX Runtime CPU (`CPUExecutionProvider`)

---

## 2. Directory Structure

```text
autovi-edge/
├── app.py                     # Responsive web inspection dashboard & MJPEG stream
├── self_test.py               # Automated commissioning self-test & sanity checker
├── install.sh                 # One-click Raspberry Pi OS installer script (PCIe Gen 3 & systemd)
├── start.sh                   # Production offline startup launcher script
├── configs/
│   ├── operational_thresholds.json   # Calibrated thresholds & review margins
│   ├── roi_configurations.json       # Deployable Normal ROI coordinates
│   └── wideresnet50_patchcore.alls   # DFC model script with hardware ImageNet normalization
├── models/
│   ├── wideresnet50_patchcore.onnx   # Standalone ONNX model (FP32 fallback, 94.95 MB)
│   ├── hailo/                        # Reproducible DFC Docker harness & build pipelines
│   │   ├── Dockerfile.hailo_dfc      # Linux x86_64 compiler container definition
│   │   ├── compile_hailo_dfc.py      # Python SDK compiler pipeline (ClientRunner)
│   │   ├── compile_hef.sh            # Automated compilation script
│   │   ├── calibration_dataset.npy   # 150 normal training calibration samples (uint8)
│   │   └── wideresnet50_patchcore.alls # Model script with hardware ImageNet normalization
│   └── memory_banks/                 # Coreset patch feature memory banks (1536-dim)
│       ├── tank_screw.pkl            # (14643, 1536)
│       ├── pipe_clip.pkl             # (8960, 1536)
│       ├── engine_wiring.pkl         # (13107, 1536)
│       ├── pipe_staple.pkl           # (8755, 1536)
│       ├── underbody_pipes.pkl       # (7372, 1536)
│       └── underbody_screw.pkl       # (17152, 1536)
├── src/
│   ├── camera.py              # Universal camera acquirer (V4L2, Picamera2, synthetic, directory playback)
│   ├── decision_engine.py     # 3-tier verdict classifier (PASS / REVIEW / DEFECT)
│   ├── heatmap.py             # Real-time anomaly heatmap & HUD overlay generator
│   └── inference_engine.py    # Dual Hailo-8 NPU / ONNX Runtime execution engine (strict stub guard)
├── tools/
│   ├── hailo_diagnostics.sh   # 4-layer PCIe Gen 3, /dev/hailo0, and firmware validator
│   ├── rpi_hailo_smoke_test.py# Standalone NPU hardware test & latency benchmark
│   └── verify_hef_artifact.py # Binary HEF vs stub artifact auditor
└── manifest.sha256            # Cryptographic SHA256 integrity manifest
```

---

## 3. Quickstart & Installation on Raspberry Pi 5

### Step 1: Transfer Bundle to Raspberry Pi
```bash
scp -r autovi-edge/ pi@<raspberry_pi_ip>:~/
ssh pi@<raspberry_pi_ip>
cd ~/autovi-edge
```

### Step 2: Run Installer
The installer sets up system drivers and a local Python virtual environment (`$HOME/autovi_venv`):
```bash
chmod +x install.sh start.sh
./install.sh
```
**Optional Flags:**
- `./install.sh --enable-pcie-gen3`: Configure Raspberry Pi 5 PCIe Gen 3 (8.0 GT/s) in `/boot/firmware/config.txt` (requires reboot). By default, standard PCIe Gen 2 is preserved.
- `./install.sh --enable-service`: Install the `autovi-edge.service` systemd unit (kept **disabled** by default until explicitly enabled).
- `./install.sh --yes`: Unattended mode (defaults to NO for boot configuration modifications).

> [!NOTE]
> **Internet Requirements**:
> - **First-Time Setup Only**: Requires active network access to download official APT packages (`hailo-all`, `libgl1`, `pciutils`) and Python packages (`onnxruntime`, `opencv-python-headless`, `scipy`, `flask`).
> - **Post-Installation (Factory Run)**: **100% OFFLINE**. All model weights, configurations, and runtime dependencies reside on local storage. No external internet or cloud calls are made during inspection.

### Step 3: Commissioning Self-Test
Verify all hardware, memory banks, and decision logic:
```bash
python self_test.py
```

### Step 4: Start Inspection Server
```bash
./start.sh tank_screw auto 0 5000
```
Open your browser at `http://<raspberry_pi_ip>:5000`.

---

## 4. Operational 3-Tier Verdict Logic

| Verdict | Score Condition | Action | Badge Indicator |
| :--- | :--- | :--- | :--- |
| **PASS** | $S < 0.85 \times \tau$ | Part nominal; release down assembly line | Green HUD Badge |
| **REVIEW_REQUIRED** | $0.85 \times \tau \le S < \tau$ | Marginal feature deviation; flag for visual auditor | Amber HUD Badge |
| **DEFECT** | $S \ge \tau$ | Confirmed defect; divert to reject bin | Red HUD Badge |

---

## 5. Industrial REST API

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/inspect` | `POST` | Triggers a single synchronous frame capture and returns verdict, margin, score, and latency breakdown. |
| `/api/category` | `GET` / `POST` | Queries or dynamically switches the active category memory bank without restarting. |
| `/api/telemetry` | `GET` | Returns live FPS, backend type, and cumulative yield statistics. |
| `/video_feed` | `GET` | MJPEG video stream with blended anomaly heatmaps. |
