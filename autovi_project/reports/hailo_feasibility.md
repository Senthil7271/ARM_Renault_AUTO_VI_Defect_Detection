# Edge AI Acceleration: Hailo-8 Deployment Feasibility Report

**Target Platform**: Hailo-8 M.2 / Mini PCIe AI Acceleration Module (advertised at 26 TOPS)
**Host Architecture**: Industrial PC / ARM Embedded Host (e.g. Raspberry Pi 5, NXP i.MX8, or Intel x86 Industrial PC)
**Target Anomaly Detection Architecture**: PatchCore (WideResNet-50-2 Backbone) vs. EfficientAD

---

## 1. Architectural Feasibility Analysis

PatchCore consists of two primary computational phases:
1. **Deep Feature Extraction**: Forward pass through a convolutional neural network (WideResNet-50-2 layers `conv1` through `layer3`).
2. **Patch Aggregation & Memory Bank Comparison**: Multi-scale feature concatenation, spatial context pooling, and nearest-neighbor search against the stored normal-feature memory bank $\mathcal{M}$.

### Split Execution Model (Accelerator vs. Host)

```
[Camera Input (256x256)]
         │
         ▼
┌──────────────────────────────────────────────┐
│  HAILO-8 AI ACCELERATOR (HEF Model)          │
│  - Pretrained Backbone (WideResNet-50-2)     │
│  - Layer2 feature map output (32x32, 512-ch) │
│  - Layer3 feature map output (16x16, 1024-ch)│
└──────────────────────────────────────────────┘
         │
         │ (PCIe / USB 3.0 DMA Transfer)
         ▼
┌──────────────────────────────────────────────┐
│  HOST CPU / ACCELERATED RUNTIME              │
│  - Bilinear alignment of Layer3 -> 32x32     │
│  - Concatenation -> 1536-dim patch vectors   │
│  - Nearest-Neighbor Search vs. Memory Bank   │
│    (using FAISS / Scikit-learn / HNSW)       │
│  - Gaussian smoothing & thresholding         │
└──────────────────────────────────────────────┘
```

**Crucial Engineering Clarification**:
The complete PatchCore algorithm **cannot execute purely on the Hailo-8 NPU** because the NPU compiler is optimized for static tensor graphs (convolutions, activations, pooling, matrix multiplications). Dynamic nearest-neighbor search across variable-sized memory banks and spatial patch sorting are not native neural network operations. Therefore, the optimal architecture partitions CNN feature extraction to the Hailo NPU, and memory bank search to the host processor.

---

## 2. Technical Evaluation Pillars

### 2.1 ONNX Export & Multi-Output Support
- The WideResNet-50-2 backbone is exported to ONNX with multi-output heads targeting `layer2` and `layer3`:
  ```python
  torch.onnx.export(
      backbone_subgraph,
      dummy_input,
      "wideresnet50_backbone.onnx",
      input_names=["input"],
      output_names=["layer2_out", "layer3_out"],
      opset_version=17
  )
  ```
- **Hailo Compiler Compatibility**: Hailo Dataflow Compiler (DFC) fully supports standard ResNet/WideResNet convolutional blocks, batch norm folding, and residual skip connections.

### 2.2 Quantization Impact (INT8 Calibration)
- Hailo-8 operates in INT8 precision.
- **Risk**: PatchCore relies on delicate Euclidean distance metrics between high-dimensional continuous feature embeddings. Coarse post-training quantization can distort relative distances.
- **Mitigation**: Perform Calibration-Aware Quantization in the Hailo Model Zoo using 1,000 representative normal production images from the AutoVI training set to minimize quantization noise in intermediate feature distributions.

### 2.3 Host-Side Memory Bank & Latency Budget
- For `pipe_clip` with 175 normal training images, coreset ratio 0.10:
  - Total normal patches extracted: $175 \times 32 \times 32 = 179,200$
  - Coreset size (10%): 17,920 vectors $\times 1,536$ floats = **110 MB memory footprint**.
- On an x86/ARM host, exact L2 nearest-neighbor search for 1,024 test patches against 17,920 vectors takes **~12–25 ms** using FAISS AVX2/NEON.
- Backbone inference on Hailo-8 takes **~4–7 ms**.
- **Total Projected Latency**: **~18–32 ms per image** (equivalent to 30–55 FPS), comfortably exceeding standard industrial line rate requirements (5–15 FPS).

---

## 3. Alternative: EfficientAD for Pure Edge Deployment

If full on-accelerator execution without host memory-bank search is required in future milestones:

| Criterion | PatchCore (WideResNet-50-2) | EfficientAD (Student-Teacher) |
|:---|:---|:---|
| **Edge Hardware Execution** | Hybrid: Backbone on NPU, Memory Bank on Host | **100% on Hailo NPU** (Pure feed-forward CNN) |
| **Host Memory Usage** | ~110–250 MB per category | **< 10 MB** (no memory bank) |
| **Inference Latency** | ~25 ms | **< 5 ms** |
| **Training Speed** | Fast (Feature extraction + Coreset) | Requires backprop training (Student/Autoencoder) |
| **Industrial Baseline** | **Golden Standard** on MVTec / AutoVI | Excellent alternative for low-power edge |

### Recommendation
1. Implement and validate the PC-based PatchCore baseline first.
2. In Phase 2, compile the WideResNet-50-2 backbone to a Hailo Executable Format (`.hef`) file.
3. Accelerate host-side memory bank querying using `faiss-cpu`.
4. If host CPU overhead is too constrained on the selected carrier board, evaluate EfficientAD as a drop-in feed-forward alternative.
