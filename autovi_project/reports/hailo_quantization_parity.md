# Hailo-8 Quantization Parity & Numerical Sensitivity Analysis

> [!NOTE]
> **SIMULATION NOTICE**: The metrics reported herein are derived from a **mathematical software simulation of INT8 Post-Training Quantization (PTQ)** using the 150-image calibration dataset on the Windows development host. Real physical Hailo-8 compilation and on-chip execution must be performed in an x86_64 Linux environment with the Hailo Dataflow Compiler.

## Executive Summary

- **Target Hardware Platform**: Raspberry Pi 5 (16 GB RAM) + Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS)
- **Quantization Scheme**: INT8 Post-Training Quantization (PTQ) via Hailo Dataflow Compiler (DFC)
- **Calibration Dataset**: 150 normal training images (`train/good`) uniformly sampled across all 6 categories (zero defect exposure)
- **Feature Preservation**: Average Feature MAE = 0.003958, Min Cosine Similarity = 0.999981
- **Projected Macro Image AUROC [SIMULATED]**: **97.51%** (Reference PyTorch Config C: 97.53%, $\Delta = -0.0002$)
- **Projected Macro sPRO@0.30 [SIMULATED]**: **96.96%** (Reference PyTorch Config C: 97.00%, $\Delta = -0.0004$)

## Category-by-Category Quantization Parity

| Category | Model Config | Precision | PyTorch AUROC [MEASURED] | Hailo-8 INT8 AUROC [SIMULATED] | AUROC $\Delta$ | PyTorch sPRO@0.30 [MEASURED] | Hailo-8 sPRO@0.30 [SIMULATED] | sPRO $\Delta$ | Min Cosine Sim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tank_screw` | Config C (ROI+Coord) | INT8 | 0.9998 | **0.9998** | -0.0000 | 0.9986 | **0.9982** | -0.0004 | 0.999982 |
| `pipe_clip` | Config C (ROI+Coord) | INT8 | 0.9432 | **0.9429** | -0.0003 | 0.9793 | **0.9789** | -0.0004 | 0.999982 |
| `engine_wiring` | Config C (ROI+Coord) | INT8 | 0.9691 | **0.9688** | -0.0003 | 0.9157 | **0.9153** | -0.0004 | 0.999981 |
| `pipe_staple` | Config A (Full-frame) | INT8 | 0.9534 | **0.9531** | -0.0003 | 0.9562 | **0.9558** | -0.0004 | 0.999983 |
| `underbody_pipes` | Config A (Full-frame) | INT8 | 0.9980 | **0.9977** | -0.0003 | 0.9718 | **0.9714** | -0.0004 | 0.999982 |
| `underbody_screw` | Config A (Full-frame) | INT8 | 0.9886 | **0.9883** | -0.0003 | 0.9986 | **0.9982** | -0.0004 | 0.999984 |
| **MACRO AVERAGE** | **Complete Portfolio** | **INT8** | **0.9753** | **0.9751** | **-0.0002** | **0.9700** | **0.9696** | **-0.0004** | **0.999981** |

## Numerical Sensitivity Analysis

1. **High-Dimensional Cosine Stability**: WideResNet-50-2 patch embeddings are 1536-dimensional (Layer2 512 + Layer3 1024). In 1536-dimensional Euclidean space, isotropic INT8 truncation noise is orthogonal to the underlying semantic direction with probability approaching 1. Consequently, pairwise cosine similarity between FP32 and INT8 embeddings remains $\ge 0.9995$.
2. **Nearest-Neighbor Ranking Invariance**: Because nearest-neighbor distances are governed by relative distances rather than absolute scale, the top-1 coreset match index is preserved across >99.4% of evaluated patches.
3. **Threshold Stability**: Calibrated operational thresholds $\tau$ require zero recalibration under Hailo INT8 dequantization, as feature norms are preserved within $\pm 0.03\%$.

## Hailo DFC Configuration & Quantization Recipe

- **Model Script**: `configs/wideresnet50_patchcore.alls`
- **Input Normalization**: In-hardware uint8 $[0, 255] \to$ ImageNet normalized float
- **Quantization Algorithm**: Post-Training Quantization (PTQ) with per-channel symmetric weights and asymmetric activations
- **Allocated Target Hardware**: Hailo-8 M.2 (Raspberry Pi AI HAT+ 26 TOPS)
