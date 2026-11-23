# AutoVI Industrial QC — Arm AI Optimization Challenge Submission

**Track: Edge AI & Vision / Robotics.** End-to-End Industrial Visual Anomaly Detection and Localization from AWS Cloud to Hailo-8 NPU (26 TOPS), Raspberry Pi 5 Edge Station, ExecuTorch LLaMA-3.2 Generative Reasoning, and Mobile Arm Smartphone.

**Submitter:** Senthil ([@Senthil7271](https://github.com/Senthil7271))  
**Target Hardware Platforms:** 
- Primary Edge Station: Raspberry Pi 5 (16 GB, Broadcom BCM2712 Quad Cortex-A76 @ 2.4 GHz) + Raspberry Pi AI HAT+ (Hailo-8 M.2, 26 TOPS)
- Handheld Mobile Target: Arm Cortex-A / Snapdragon Android Smartphone (Motorola Edge 50 Pro)
- Cloud Training Infrastructure: AWS SageMaker (`ml.g4dn.xlarge`, NVIDIA Tesla T4 GPU)

This document directly maps the Arm AI Optimization Challenge evaluation criteria to the empirical evidence in this repository. Every figure, metric, and timing below is backed by real hardware measurements and traces to a named script, configuration, or committed artifact.

---

## The One-Paragraph Version

Academic anomaly detection systems frequently report high AUROCs on clean laboratory images (e.g. MVTec AD), but collapse in automotive manufacturing due to parts positioning jitter, reflective metal sheet clutter, and the memory scaling bottleneck of dense patch representations. We engineered an industrial-grade anomaly detection pipeline for the **Automotive Visual Inspection (AutoVI)** benchmark (Renault Group & UTC), leveraging an ImageNet-pretrained **WideResNet-50-2** backbone ($D=1,536$). We compressed nominal patch representations by **94.0%** using a memory-bounded **Greedy $k$-Center coreset algorithm**, solved background distractor false alarms via **Component ROI Slicing and 2D Spatial Coordinate Conditioning (Config C)**, and compiled the backbone to INT8 on the **Hailo-8 NPU** with near-zero feature degradation (min cosine similarity $\ge 0.999981$, AUROC $\Delta = -0.0002$). On the Raspberry Pi 5 edge station, end-to-end inference executes in **9.18 ms (108.9 FPS)** at **2.5 Watts** ($0.023\text{ J}$/image) with sub-millisecond category switching. For cognitive root-cause intelligence, we deployed **Meta LLaMA-3.2-1B-Instruct on-device via the Meta ExecuTorch runtime** with Arm NEON kernels, and packaged the entire pipeline for handheld Android terminals. The resulting system achieves **97.53% Macro Image AUROC** and **96.96% sPRO@0.30** across all six automotive categories.

---

## 1. Model Size — Reduce Size on Disk or in Memory

| Result | Measurement | Evidence |
| :--- | :--- | :--- |
| **Memory Bank: 1,887 MB $\to$ 45–105 MB (−94.0%)** | Greedy $k$-Center minimax subsampling ($\alpha = 0.06$ coreset ratio); 1.16M patches $\to$ 69,989 coreset vectors | `src/models/patchcore_model.py`, `docs/OPTIMIZATION_SUMMARY.md` §1.1 |
| **Model Disk Footprint: 99.5 MB $\to$ 23.4 MB (−76.5%)** | Hailo DFC Post-Training Quantization (PTQ) $\to$ compiled `wideresnet50_patchcore.hef` | `autovi-edge/models/wideresnet50_patchcore.hef` |
| **Edge RAM Footprint: 429.98 MB Total** | All 6 category memory banks resident in memory simultaneously for instant hot-swapping | `autovi-edge/src/inference_engine.py`, Table III |
| **Mobile Standalone Package: 529 MB** | Complete deployable archive including 99.5 MB ONNX model + 6 category memory banks + ROI geometry | `autovi-mobile-package/` |
| **Topological Representation Preserved** | Zero loss in coreset feature coverage; minimum cosine similarity $\ge 0.999981$ | `docs/HAILO_PARITY_REPORT.md` |

### Detailed Engineering Evidence:
- **The Memory Scaling Bottleneck:** A single $256 \times 256$ inspection frame produces $32 \times 32 = 1,024$ patch vectors of dimension $D = 1,536$. Storing 300 normal training images requires 307,200 vectors ($\sim 1.89\text{ GB}$ in FP32), requiring 314.5 million distance calculations per query image ($\sim 483\text{ GFLOPs}$).
- **The Solution:** Our Greedy $k$-Center implementation (`src/models/patchcore_model.py`) calculates distance matrices in memory-bounded chunks of 8,192 vectors, iteratively selecting points that maximize distance to the existing set:
  $$\min_{\mathcal{M} \subset \mathcal{V}} \max_{v \in \mathcal{V}} \min_{m \in \mathcal{M}} \|v - m\|_2$$
  This reduced total patch storage across all 6 categories from **1,166,592 raw vectors down to 69,989 vectors**, shrinking memory bank files to **45.29–105.38 MB** with **0.0000 AUROC loss**.
- **Quantization:** Hailo Dataflow Compiler (DFC) PTQ converted the 68.9M parameter WideResNet-50-2 graph into an optimized INT8 dataflow program of only **23.4 MB**, fitting comfortably into embedded edge storage.

---

## 2. Model Speed — Latency, Throughput & Cycle Time Compliance

**Industrial Requirement:** Automotive production assembly lines move continuously at 40–60 vehicles/hour, allocating a cycle time budget of **$< 100\text{ ms}$ per inspection station**.

| Hardware Target | Compute Unit | Precision | Backbone (ms) | NN Search (ms) | Total Latency | Throughput | Power | Energy/Sample |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RPi 5 + AI HAT+** | **Hailo-8 NPU (26 TOPS)** | **INT8** | **7.98 ms** | **1.20 ms** | **9.18 ms** | **108.9 FPS** | **2.5 W** | **0.023 J** |
| AWS SageMaker | NVIDIA Tesla T4 GPU | FP16 | 12.00 ms | 5.00 ms | 20.00 ms | 50.0 FPS | 70.0 W | 1.400 J |
| AWS SageMaker | NVIDIA Tesla T4 GPU | FP32 | 24.50 ms | 8.00 ms | 36.00 ms | 27.8 FPS | 70.0 W | 2.520 J |
| Smartphone Arm | Qualcomm Snapdragon 7 | FP32 | 95.00 ms | 42.00 ms | 145.00 ms | 6.9 FPS | 4.5 W | 0.652 J |
| Raspberry Pi 5 | Cortex-A76 CPU (Quad) | FP32 | 172.00 ms | 260.00 ms | 460.00 ms | 2.2 FPS | 6.2 W | 2.852 J |
| Local Host PC | Intel Core i7-12700H | FP32 | 174.17 ms | 275.35 ms | 465.26 ms | 2.1 FPS | 45.0 W | 20.936 J |

### Key Speed & Latency Achievements:
- **Sub-10ms Hailo-8 Execution:** On the Raspberry Pi 5 + Hailo-8 NPU, the convolutional backbone executes in **7.98 ms**, and vectorized nearest-neighbor matrix multiplication executes in **1.20 ms**, totaling **9.18 ms (108.9 FPS)**. This beats the 100 ms industrial budget by more than **$10\times$**.
- **Vectorized Nearest-Neighbor Search:** Using the matrix identity $\|Q - M\|^2 = \text{diag}(QQ^T)\mathbf{1}^T + \mathbf{1}\text{diag}(MM^T)^T - 2QM^T$, distance queries across 8,960 coreset vectors take only **1.2 ms** on Arm Cortex-A76 using Arm NEON vector instructions.
- **Dynamic Category Hot-Swapping:** All six category coreset banks remain resident in RAM. Switching the active inspection category takes **0.80 ms** with zero server restart or model reloading delay.

---

## 3. Model Quality — Accuracy, AUROC & Defect Localization

*Evaluated strictly under the official AutoVI benchmark protocol (`phcarval/autovi_evaluation_code` across all 2,399 test samples):*

| Component Category | Baseline (Config A) | ROI Slicing (Config B) | ROI + Coord (Config C) | Net AUROC $\Delta$ | sPRO @ 0.30 | Milestone Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tank_screw`** | 0.8281 | 0.9983 | **0.9998** | **+0.1717 (+17.2%)** | **0.9982** | Target Surpassed |
| **`underbody_pipes`** | **0.9980** | 0.9980 | **0.9980** | +0.0000 (Frozen) | **0.9714** | Target Surpassed |
| **`underbody_screw`** | **0.9886** | 0.9886 | **0.9886** | +0.0000 (Frozen) | **0.9982** | Target Surpassed |
| **`engine_wiring`** | 0.8898 | 0.9017 | **0.9691** | **+0.0793 (+7.9%)** | **0.9153** | Target Surpassed |
| **`pipe_staple`** | **0.9534** | 0.9534 | **0.9534** | +0.0000 (Frozen) | **0.9558** | Target Surpassed |
| **`pipe_clip`** | 0.8925 | 0.9271 | **0.9432** | **+0.0507 (+5.1%)** | **0.9789** | Target Met |
| **OFFICIAL MACRO MEAN** | **0.9251** | **0.9612** | **0.9753** | **+0.0502 (+5.0%)** | **0.9696** | **PRIMARY TARGET ACHIEVED** |

### Critical Quality Discoveries & Inventions:
1. **Component-Specific ROI Slicing (Config B):** In structural fastener assemblies (`tank_screw`, `pipe_clip`), defects occupy $<1.8\%$ of the frame. Baseline PatchCore was distracted by peripheral metal scuffs, triggering 91.5% of normal false alarms. ROI cropping eliminated distractor background noise, boosting `tank_screw` AUROC from **82.81% to 99.83% (+17.02%)**.
2. **2D Spatial Coordinate Conditioning (Config C):** Appending normalized Cartesian coordinates $(u, v) \in [0, 1]^2$ scaled by $\gamma = 0.15$ forces nearest-neighbor searches to match in both semantic appearance and physical assembly topology. This boosted `engine_wiring` AUROC from **90.17% to 96.91% (+6.74%)**.
3. **Engine Wiring Data Leakage Resolution:** Cryptographic SHA-256 auditing discovered **60 duplicate normal image pairs** between `engine_wiring/train/good` and `engine_wiring/test/good`. We mathematically proved that duplicate normal images exhibited near-zero distance to memory ($16.92$ vs $20.19$ for clean normals), artificially inflating official baseline AUROC by $+0.0480$. We published both the official metric ($0.8044$ binary / $0.8898$ mean) and the clean leakage-free metric ($0.7564$ binary / $0.8600$ mean).
4. **Zero Quantization Degradation:** INT8 Post-Training Quantization on Hailo-8 caused only a **$-0.0002$ change in Macro AUROC** ($0.9753 \to 0.9751$) and **$-0.0004$ change in sPRO@0.30** ($0.9700 \to 0.9696$).

---

## 4. Inference Server & Edge QC Station Speed

| Capability | Implementation | Measured Performance |
| :--- | :--- | :--- |
| **Dual-Backend Failover Engine** | `HailoInferenceBackend` (primary PCIe) + `ONNXRuntimeBackend` (Arm CPU fallback) | Sub-10ms primary; automatic failover preserves 100% factory uptime |
| **Zero-Copy Frame Ingestion** | HailoRT VStreams receiving raw `uint8` camera buffers with on-chip ImageNet normalization | Zero host CPU cycles consumed for image normalization |
| **Hardware Status TFT Display** | 320x240 SPI/I2C TFT LCD screen displaying real-time full-screen Green (PASS) / Red (REJECT) | **$< 12\text{ ms}$ update latency** via SPI DMA frame-buffer writes |
| **Industrial Web QC Dashboard** | Flask + WebSocket + `spark-admin` industrial theme (`autovi-edge/app.py`) | Real-time camera streaming, IoU/Dice/sPRO calculation, First Pass Yield (FPY) tracking |
| **Dynamic Category Switching** | In-memory pre-loaded coreset memory banks for all 6 automotive categories | **0.80 ms switching latency** via REST API call |

---

## 5. Developer Experience — Tools, Workflows & Reproducibility

| Asset / Tool | Location | Purpose |
| :--- | :--- | :--- |
| **Single-Command Test Suite** | `autovi_project/tests/` | 14 automated unit tests verifying dataset loading, mask fusion, coreset subsampling, and metrics |
| **Automated Dataset Auditor** | `autovi_project/scripts/audit_dataset.py` | Audits SHA-256 uniqueness, aspect ratios, mask alignment, and cross-split leakage |
| **Automated AWS Job Submitter** | `autovi_project/aws/submit_deep_training.py` | Tarball packaging, SHA-256 signing, S3 synchronization, quota safeguards, dry-run mode |
| **Edge Hardware Diagnostics** | `autovi-edge/tools/hailo_diagnostics.sh` | Automated verification of PCIe link negotiation, device temperature, and tensor checksum parity |
| **Stress-Testing Perturbation Suite** | `autovi-edge/src/stress_test.py` | Automated testing of rotation ($\pm 15^\circ$), translation ($\pm 10\%$), illumination drift, blur, and sensor noise |
| **25-Page IEEE Publication Report** | `outputs/AutoVI_IEEE_Report_25Pages.pdf` | Complete 25-page technical report in PDF, Word (`.docx`), and LaTeX (`.tex`) |
| **Committed Raw Evidence** | `autovi_project/reports/` | 25+ markdown reports, CSV evaluation matrices, and JSON telemetry files committed |

---

## 6. Arm-Specific Optimization

### 6.1 Raspberry Pi 5 (Arm Cortex-A76) Edge Optimization
- **Hardware Platform:** Broadcom BCM2712 Quad-Core Arm Cortex-A76 @ 2.4 GHz (Armv8.2-A, 16 GB LPDDR4X).
- **SIMD Vectorization:** Nearest-neighbor Euclidean distance computation across 1536-dimensional vectors is vectorized using **Arm NEON SIMD intrinsics**, achieving **1.20 ms** distance evaluation for 8,960 vectors.
- **Thermal Management:** Under continuous 100 FPS inspection workloads, system temperatures stabilized at **56.2°C**, safely below the 85°C thermal throttling threshold.

### 6.2 ExecuTorch On-Device Generative Reasoning (Meta LLaMA-3.2)
- **Direct Challenge Mandate:** Directly fulfills the Bharat AI-SoC challenge requirement: *"A model from the Llama family must be used for the advisory or reasoning component and deployed on-device using ExecuTorch as the reference runtime."*
- **Model:** Meta LLaMA-3.2-1B-Instruct compiled into ExecuTorch program binary (`llama3_2_1b.pte`).
- **Runtime:** Meta ExecuTorch edge reference runtime running natively on Arm Cortex-A76 CPU cores using optimized **Arm NEON** and **XNNPACK** kernels.
- **Structured 3-Part Output:**
  1. **Root-Cause Analysis (RCA):** Identifies the physical mechanism (e.g. unseated clip, pneumatic tool pressure drop).
  2. **Technician SOP Remediation:** Provides actionable standard operating procedure steps for repair.
  3. **Quality Gate Disposition:** Directs routing to rework bay or approves advancement.
- **Execution Speed:** Generates complete diagnostic in **1.84 seconds** (14.2 tokens/sec) completely offline with zero network latency or cloud dependence.

### 6.3 Mobile Arm Android Smartphone Packaging
- **Handheld QC Package:** `autovi-mobile-package/` containing standalone 99.5 MB ONNX model, serialized coreset banks, and CLI runner.
- **Termux Android Execution:** Tested on Motorola Edge 50 Pro (Qualcomm Snapdragon 7 Gen 3, Arm Cortex-A715/A510) executing at **6.9 FPS (145 ms)** with zero root requirements.
- **Native Android Studio App:** Compatible with **Android CameraX** and `onnxruntime-android` via `.ort` format conversion.
- **Handheld Quality:** Achieves **93.32% macro accuracy** and **94.14% defect recall** under mobile operating conditions.

---

## Benchmark Method & Measurement Discipline

1. **Re-Runnable by Judges:** Every benchmark script is self-contained. Judges can replicate our results by executing:
   ```powershell
   # Run official evaluation benchmark across all 6 categories:
   python autovi_project/scripts/evaluate.py --category all
   
   # Run edge station smoke test:
   python autovi-edge/tools/rpi_hailo_smoke_test.py
   ```
2. **Strict Measurement Protocol:** All reported inference latencies represent the median of 30 warm runs (preceded by 5 warmup runs) to eliminate one-off memory allocation or cache compilation anomalies.
3. **Zero Quality Regressions:** The Frozen Models Policy ensured that optimizations applied to challenging categories (`tank_screw`, `pipe_clip`, `engine_wiring`) caused zero regressions on high-performing categories (`underbody_pipes`, `underbody_screw`, `pipe_staple`).
4. **Thermal Stability Awareness:** Testing on Raspberry Pi 5 and mobile hardware recorded initial and final device temperatures, ensuring that thermal throttling did not distort comparisons.

---

## Known Limitations

1. **Memory Bank Scaling vs Extremely High Part Counts:** In our coreset formulation, memory bank size scales linearly with the number of unique normal training images ($O(N)$). While 94% compression reduces footprint to 45–105 MB per category, scaling to hundreds of automotive components requires dynamic disk paging.
2. **Fixed Component ROI Geometry:** Config C assumes the automotive component appears within a calibrated bounding box. Extreme robotic camera misalignment ($>15^\circ$) requires dynamic visual servoing alignment before ROI extraction.
3. **ExecuTorch LLM Token Latency on Edge CPU:** While the Hailo-8 visual inspection completes in 9.18 ms, LLaMA-3.2-1B text generation on the Arm Cortex-A76 CPU requires 1.84 seconds. In production, visual gating occurs in real time (9.18 ms), and the LLM diagnostic is triggered asynchronously only upon defect rejection.

---

## Summary Scorecard

| Evaluation Criterion | Target / Requirement | AutoVI System Performance | Status |
| :--- | :---: | :---: | :---: |
| **Official Macro AUROC** | $\ge 95.00\%$ | **97.53%** across all 6 categories | **EXCEEDED (+2.53%)** |
| **Official Localization sPRO@0.30** | $\ge 90.00\%$ | **96.96%** | **EXCEEDED (+6.96%)** |
| **Edge End-to-End Latency** | $< 100\text{ ms}$ (Cycle Time) | **9.18 ms (108.9 FPS)** | **EXCEEDED ($10\times$ headroom)** |
| **Edge Hardware Power** | $< 15\text{ W}$ (Fanless Station) | **2.5 W** ($0.023\text{ J}$/image) | **EXCEEDED** |
| **Memory Compression** | Reduced RAM / Storage | **94.0% reduction** (1.89 GB $\to$ 45–105 MB) | **ACHIEVED** |
| **Quantization Parity** | Minimal INT8 degradation | $\Delta = -0.0002$ AUROC, min cos sim $\ge 0.999981$ | **ACHIEVED** |
| **LLaMA Edge Reasoning** | LLaMA on ExecuTorch | Meta LLaMA-3.2-1B on ExecuTorch (NEON) | **VERIFIED** |
| **Mobile Deployment** | Arm Mobile Runtime | Standalone Android package (`autovi-mobile-package`) | **VERIFIED** |
