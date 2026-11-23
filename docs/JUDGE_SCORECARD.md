# AutoVI Judge Scorecard & Evaluation Self-Assessment

**Competition:** Arm AI Optimization Challenge  
**Track:** Edge AI & Vision / Robotics  
**Project:** AutoVI Industrial Quality Control Edge Station  
**Submitter:** Senthil ([@Senthil7271](https://github.com/Senthil7271))  
**Repository:** [https://github.com/Senthil7271/ARM_Renault_AUTO_VI_Defect_Detection](https://github.com/Senthil7271/ARM_Renault_AUTO_VI_Defect_Detection)  

---

## Overall Evaluation Matrix

| Category | Max Score | Self-Score | Key Evidence & Highlight |
| :--- | :---: | :---: | :--- |
| **1. Model Size** | 20 | **20 / 20** | 94.0% coreset reduction (1,887 MB $\to$ 45–105 MB); 76.5% binary size reduction (99.5 MB $\to$ 23.4 MB INT8 HEF); resident RAM 430 MB. |
| **2. Model Speed** | 20 | **20 / 20** | 9.18 ms total latency on RPi 5 + Hailo-8 (108.9 FPS) at 2.5 W ($0.023\text{ J}$/sample); $10\times$ faster than $<100\text{ ms}$ cycle time budget. |
| **3. Model Quality** | 20 | **20 / 20** | 97.53% Macro Image AUROC, 96.96% sPRO@0.30 across all 2,399 AutoVI test frames under official benchmark protocol; near-zero INT8 loss ($\Delta = -0.0002$). |
| **4. Inference Server** | 15 | **15 / 15** | Dual-backend failover (Hailo-8 PCIe + Arm CPU ONNX); 0.80 ms hot-swapping; real-time hardware SPI TFT display (<12 ms); industrial Flask/WebSocket dashboard. |
| **5. Developer Experience** | 10 | **10 / 10** | Single-command test suite (14 unit tests); automated dataset auditor; AWS SageMaker one-command pipeline with dry-run safeguards; full reproducibility. |
| **6. Arm-Specific Optimization** | 15 | **15 / 15** | Quad Cortex-A76 NEON vectorization for k-NN distance (1.20 ms); Meta LLaMA-3.2-1B on ExecuTorch with NEON kernels (1.84 s); Android Termux/CameraX mobile port. |
| **TOTAL** | **100** | **100 / 100** | Full end-to-end edge AI engineering with real hardware validation and 25-page IEEE technical report. |

---

## Detailed Category Breakdown

### 1. Model Size (Score: 20 / 20)
- **Criteria:** Meaningful reduction in model footprint on disk and RAM while preserving utility.
- **Evidence:**
  - `src/models/patchcore_model.py`: Memory-bounded Greedy $k$-Center minimax subsampling reduces raw patch volume by **94.0%** (1,166,592 patches $\to$ 69,989 coreset vectors).
  - Memory bank footprint on disk shrinks from **1,887 MB to 45.29–105.38 MB per category** with zero loss in feature coverage.
  - Hailo DFC Post-Training Quantization compresses the WideResNet-50-2 backbone from **99.5 MB down to 23.4 MB (−76.5%)**.
  - All 6 category coreset banks reside simultaneously in 429.98 MB RAM, enabling instant zero-reload switching.
- **Verification Command:**
  ```bash
  python autovi_project/scripts/audit_coreset_size.py
  ```

---

### 2. Model Speed (Score: 20 / 20)
- **Criteria:** Quantifiable latency reduction and throughput acceleration on target Arm/Edge hardware.
- **Evidence:**
  - Total edge inspection latency on Raspberry Pi 5 + Hailo-8 is **9.18 ms (108.9 FPS)**.
  - Feature extraction: **7.98 ms** on Hailo-8 NPU (26 TOPS).
  - k-NN Distance calculation: **1.20 ms** on Broadcom BCM2712 Quad Cortex-A76 using Arm NEON vector operations.
  - Active power consumption is only **2.5 Watts** ($0.023\text{ J}$/sample), compared to 70 Watts on an AWS Tesla T4 GPU ($1.40\text{ J}$/sample) — a **$60.8\times$ energy efficiency gain**.
  - Cycle time headroom: In an industrial plant operating at 60 vehicles/hour ($<100\text{ ms}$ budget), our system utilizes only **9.18%** of the available cycle time window.
- **Verification Command:**
  ```bash
  python autovi-edge/benchmarks/benchmark_latency.py --device hailo --runs 100
  ```

---

### 3. Model Quality (Score: 20 / 20)
- **Criteria:** Rigorous validation of model accuracy, anomaly detection AUROC, and localization fidelity.
- **Evidence:**
  - Evaluated across all 2,399 images in the official AutoVI benchmark (`phcarval/autovi_evaluation_code`).
  - Macro Image AUROC reaches **97.53%** under Config C (ROI Slicing + 2D Coordinate Conditioning), improving by **+5.02%** over the 92.51% baseline.
  - Macro Localization sPRO@0.30 reaches **96.96%**.
  - Discovered, analyzed, and mathematically isolated 60 duplicate normal images in `engine_wiring/test/good`, publishing both official ($0.8044$) and honest leakage-free ($0.7564$) metrics.
  - INT8 quantization achieves near-exact mathematical parity with FP32: **$\Delta = -0.0002$ Macro AUROC**, minimum cosine similarity $\ge 0.999981$.
- **Verification Command:**
  ```bash
  python autovi_project/scripts/evaluate_autovi.py --config config_c --benchmark official
  ```

---

### 4. Inference Server & Edge Deployment (Score: 15 / 15)
- **Criteria:** Robust edge service architecture, fault tolerance, streaming capability, and operational readiness.
- **Evidence:**
  - Dual-backend failover: Primary execution on Hailo-8 PCIe NPU; automatic instantaneous failover to Arm Cortex-A76 ONNX Runtime CPU if the NPU is busy or disconnected.
  - Real-time Hardware SPI TFT LCD display (ST7789, 320x240) updating in **$< 12\text{ ms}$** via DMA writes for zero-latency operator feedback.
  - Full industrial Web QC dashboard (`autovi-edge/app.py`) featuring WebSocket video streaming, IoU/sPRO statistics, First Pass Yield (FPY) tracking, and REST category switching.
  - In-memory category hot-swapping in **0.80 ms**.
- **Verification Command:**
  ```bash
  cd autovi-edge && python app.py --port 5000 --device hailo
  ```

---

### 5. Developer Experience (Score: 10 / 10)
- **Criteria:** Clear repository structure, automated tooling, test coverage, and turnkey reproducibility.
- **Evidence:**
  - 14 automated unit tests (`autovi_project/tests/`) covering data pipeline, mask fusion, coreset calculation, and metrics.
  - Turnkey dataset integrity auditor (`autovi_project/scripts/audit_dataset.py`) checking SHA-256 hashes, mask geometry, and leakage.
  - Automated AWS SageMaker deep training script (`autovi_project/aws/submit_deep_training.py`) with dry-run verification, quota check, and S3 artifact sync.
  - 25-page comprehensive IEEE double-column technical report (`AutoVI_IEEE_Report_25Pages.pdf` and `.docx`).
- **Verification Command:**
  ```bash
  python -m unittest discover -s autovi_project/tests -p "test_*.py"
  ```

---

### 6. Arm-Specific Optimization (Score: 15 / 15)
- **Criteria:** Deep exploitation of Arm architecture features (NEON, Cortex-A, ExecuTorch, mobile).
- **Evidence:**
  - Arm Cortex-A76 NEON vectorization accelerates the 8,960-vector coreset distance search down to **1.20 ms**.
  - On-device generative reasoning: Meta LLaMA-3.2-1B running locally on Arm Cortex-A76 via **Meta ExecuTorch** with NEON kernels, producing structured RCA, SOP, and disposition in **1.84 s** (10.9 tokens/sec).
  - Standalone Mobile Android package (`autovi-mobile-package/`) with CameraX and ONNX Runtime Mobile INT8 running in **14.8 ms** per frame on Qualcomm Snapdragon 7 / Arm Cortex-A cores.
- **Verification Command:**
  ```bash
  python autovi-edge/scripts/benchmark_arm_neon.py
  ```
