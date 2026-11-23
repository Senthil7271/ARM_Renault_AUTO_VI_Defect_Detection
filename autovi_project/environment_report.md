# Environment Audit Report

**Date & Time**: 2026-11-14
**Operating System**: Windows (AMD64)
**Workspace Root**: `C:\Users\linga\Downloads\ArmChallenge`

## 1. Runtime & Package Inspection

| Component | Detected State | Notes |
|:---|:---|:---|
| **Python Version** | 3.10.11 (64-bit) | Located at `C:\Users\linga\AppData\Local\Programs\Python\Python310\python.exe` |
| **PyTorch** | `2.12.1+cpu` | Functional CPU build |
| **CUDA / GPU** | `False` | No active NVIDIA GPU / driver interface detected |
| **TorchVision** | `0.27.1+cpu` | Pretrained vision backbones supported |
| **NumPy** | `2.2.6` | Verified compatible |
| **SciPy** | `1.15.3` | Spatial distance & KD-tree utilities |
| **scikit-learn** | `1.7.2` | ROC, PR, confusion matrix, nearest neighbors |
| **OpenCV** | `4.10.0` | Image/mask I/O and spatial processing |
| **Pillow (PIL)** | `12.2.0` | Image handling and visualization |
| **Pandas** | `2.3.3` | Audit reporting and tabular metrics |
| **Matplotlib** | `3.10.9` | Plotting curves and four-panel inspection overlays |
| **PyYAML** | `6.0.2` | Configuration file parsing |

## 2. Hardware Constraints & Workarounds
- **Execution Target**: PyTorch CPU mode with multi-threading (`torch.set_num_threads`).
- **DataLoader Workers**: Set to `num_workers = 0` to prevent Windows IPC and multiprocessing spawn overhead/deadlocks.
- **Memory Optimization**: PatchCore coreset subsampling ratio is configurable (`0.10` by default) with random projection dimensionality reduction to keep memory-bank computation lightweight and memory-efficient on CPU.
- **Pretrained Weights**: WideResNet-50-2 is loaded via torchvision weights. In offline scenarios, weights can be cached locally or passed via a checkpoint path.
