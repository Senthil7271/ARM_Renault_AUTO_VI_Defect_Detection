# AutoVI: Mobile & Embedded Arm Industrial Inspection Package

This package contains the minimal, standalone runtime files required to deploy and evaluate the **AutoVI PatchCore Industrial Anomaly Detection Engine** on **Arm-based mobile and edge devices** (Android smartphones, Raspberry Pi, and Cortex-A/Snapdragon embedded platforms).

---

## 1. Directory Structure

```
autovi-mobile-package/
├── models/
│   ├── wideresnet50_patchcore.onnx   # Production ONNX FP32 model (99.5 MB)
│   ├── export_pytorch_model.py       # Script to trace wideresnet50_patchcore.pt
│   └── memory_banks/                 # Coreset memory banks with calibrated thresholds
│       ├── pipe_clip.pkl             # tau = 16.75 (85.8% Acc, 85.2% Recall)
│       ├── pipe_staple.pkl           # tau = 17.28 (100.0% Acc, 100.0% Recall)
│       ├── engine_wiring.pkl         # tau = 17.85 (85.5% Acc, 85.7% Recall)
│       ├── tank_screw.pkl            # tau = 16.50 (98.5% Acc, 100.0% Recall)
│       ├── underbody_pipes.pkl       # tau = 21.34 (92.2% Acc, 99.5% Recall)
│       └── underbody_screw.pkl       # tau = 14.44 (98.0% Acc, 94.4% Recall)
├── configs/
│   └── roi_configurations.json       # Deployable Component-ROI bounding boxes
├── mobile_infer.py                   # Mobile inspection CLI (supports ONNX & PyTorch)
└── README.md                         # Package documentation & threshold reference
```

---

## 2. Model Specifications

* **Backbone Architecture**: WideResNet-50-2 (Pre-trained on ImageNet-1K).
* **Multi-Scale Neighborhood Aggregation**:
  * Layer 2 (512-dim, $32\times 32$) + Layer 3 (1024-dim, $16\times 16$ upsampled $2\times$ to $32\times 32$).
  * Aggregated feature dimension: **$1,536$ channels** per spatial patch.
* **Input Tensor**:
  * Shape: `[1, 3, 256, 256]` (Float32, RGB order).
  * Normalization: ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`.
* **Output Tensor**:
  * Shape: `[1, 1536, 32, 32]`.
* **Anomaly Scoring Algorithm**:
  * Config C: Deployable Normal ROI + 2D Gaussian Spatial Coordinate Conditioning + $5\times 5$ Gaussian Smoothing ($\sigma = 1.2$).
  * Decision Rule: `Score >= tau` $\rightarrow$ **REJECT [DEFECT]**; `Score < tau` $\rightarrow$ **PASS [OK]**.

---

## 3. Calibrated Operational Threshold Reference ($\tau$)

| Category | Input Res (px) | Calibrated $\tau$ | Accuracy | Defect Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`pipe_staple`** | $400 \times 400$ | **`17.28`** | **`100.00%`** | `100.00%` | Flawless |
| **`tank_screw`** | $1000 \times 750$ | **`16.50`** | **`98.50%`** | `100.00%` | High Confidence |
| **`underbody_screw`**| $1000 \times 750$ | **`14.44`** | **`98.00%`** | `94.44%` | High Confidence |
| **`underbody_pipes`**| $1000 \times 750$ | **`21.34`** | **`92.17%`** | `99.46%` | High Confidence |
| **`pipe_clip`** | $400 \times 400$ | **`16.75`** | **`85.76%`** | `85.21%` | Target Met |
| **`engine_wiring`** | $400 \times 400$ | **`17.85`** | **`85.50%`** | `85.71%` | Target Met |
| **OVERALL PORTFOLIO**| — | — | **`93.32%`** | **`94.14%`** | **Industrial Grade** |

---

## 4. Mobile Execution Instructions

### A. Run via Termux on Android Phone (Motorola Edge 50 Pro / Any Arm Phone)
1. Install **Termux** from F-Droid.
2. Install Python and ONNX Runtime:
   ```bash
   pkg update && pkg install python opencv clang -y
   pip install numpy opencv-python onnxruntime
   ```
3. Copy this folder to the phone storage and run:
   ```bash
   # Inspect a single image:
   python3 mobile_infer.py --category pipe_clip --image test_sample.jpg

   # Run live camera inspection:
   python3 mobile_infer.py --category pipe_staple --camera 0
   ```

### B. Run with PyTorch FP32
If your mobile environment has PyTorch installed (`pip install torch torchvision`):
1. (Optional) Generate the TorchScript `.pt` binary:
   ```bash
   python3 models/export_pytorch_model.py
   ```
2. Run inspection with PyTorch backend:
   ```bash
   python3 mobile_infer.py --category pipe_clip --image test_sample.jpg --backend pytorch
   ```

### C. Native Android App (CameraX + ONNX Runtime Mobile)
1. Convert ONNX to optimized Mobile ORT format:
   ```bash
   python3 -m onnxruntime.tools.convert_onnx_models_to_ort models/wideresnet50_patchcore.onnx
   ```
2. Import `wideresnet50_patchcore.ort` into your Android Studio `app/src/main/assets/`.
3. Add dependency to `build.gradle`:
   ```groovy
   implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'
   ```
4. Feed camera frames directly to the model using Arm NEON hardware acceleration.
