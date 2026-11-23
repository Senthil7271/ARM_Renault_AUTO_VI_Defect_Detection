# AutoVI Phase 3 — Deep Anomaly Representation Learning
## Research Audit, Architectural Candidates, and GPU Training Plan

**Status**: Gate A Complete (Local Engineering & Zero-Cost Validation)  
**Authors**: Senior AWS ML Infrastructure Engineer & Industrial Anomaly Detection Researcher  
**Target Platform**: AWS SageMaker `ml.g4dn.xlarge` (NVIDIA Tesla T4 GPU, 16 GiB VRAM)  
**Priority Categories**: `pipe_clip`, `tank_screw`, `engine_wiring`  
**Reference Checkpoints**: Preserved (AWS SageMaker Run `autovi-all-2026-11-15-06-03-24`)  

---

## 1. Executive Summary & Research Objectives

In Phase 1 and Phase 2, we established the official AutoVI benchmark across all six product categories using memory-bounded PatchCore with WideResNet-50-2. Phase 2 optimized inference scoring (demonstrating that Top-9 Mean patch distance outperforms raw max and reference reweighting).

However, three categories remain performance-constrained due to fundamental representation limits of frozen generic ImageNet features:
1. **`pipe_clip`**: Subtle mechanical defects (`unclipped` brake pipe) differ only slightly from normal variants. Calibrated recall remains constrained at **$31.7\%$** (Top-9 Mean).
2. **`tank_screw`**: Small fasteners on fuel tanks with high background variance. Calibrated recall is **$41.1\%$** with a **$13.5\%$** false alarm rate.
3. **`engine_wiring`**: High topological variance in wire harness routing. Calibrated recall is **$45.0\%$** (Top-9 Mean), with **60 duplicated normal test images** that artificially inflate standard benchmark AUROC from $0.7564$ to $0.8044$.

**Objective**: Upgrade these three categories using **trainable deep anomaly representations** that learn normal industrial feature distributions and distinguish subtle mechanical anomalies from nominal parts.

---

## 2. Preserved Baselines & Performance Benchmarks

All existing six PatchCore WideResNet-50-2 checkpoints, official evaluation outputs, and Phase 2 experiments remain strictly preserved.

### 2.1 Baseline Metrics for the Three Priority Categories

| Category | Preserved Model Checkpoint | Feature Dim ($D$) | Memory Bank Shape | Pooled Image AUROC (Raw Max) | Pooled Image AUROC (Top-9 Mean) | Official Mean Image AUROC | Official AUC-sPRO @ 0.30 | Baseline Calibrated Recall | Baseline False Alarm Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pipe_clip`** | `WideResNet-50-2` (68.9M params) | 1,536 | $(8,960 \times 1,536)$ | 0.7515 | **0.7778** | **0.8925** | 0.9525 | 31.7% | 15.4% |
| **`tank_screw`** | `WideResNet-50-2` (68.9M params) | 1,536 | $(14,643 \times 1,536)$ | 0.7927 | **0.8162** | **0.8281** | 0.9931 | 41.1% | 13.5% |
| **`engine_wiring`** | `WideResNet-50-2` (68.9M params) | 1,536 | $(13,107 \times 1,536)$ | 0.8044 | **0.8210** | **0.8898** | 0.8066 | 45.0% | 7.0% |

### 2.2 Leakage-Aware Clean Baseline for `engine_wiring`
- **Official Split ($N=607$)**: 285 normal, 322 defective $\rightarrow$ Pooled AUROC = **0.8044** (Top-9: **0.8210**).
- **Clean Split ($N=547$)**: 225 clean normal, 322 defective $\rightarrow$ Pooled AUROC = **0.7564** (Top-9: **0.7767**).
- **Evaluation Rule**: The 60 duplicate normal images will continue to be documented separately; no test normal images will be used in model fitting or checkpoint selection.

---

## 3. Trainable Architectural Candidates

We have implemented two complementary deep anomaly representation architectures in `autovi_project/src/models/`:

### 3.1 Candidate A — Reverse Distillation (RD)
- **File**: `src/models/reverse_distillation.py`
- **Theory (Deng & Li, CVPR 2022)**:
  - **Teacher**: Pretrained `WideResNet-50-2` (or `ResNet-50`) multi-scale intermediate activations (`layer1`, `layer2`, `layer3`), strictly frozen (`requires_grad = False`).
  - **One-Class Embedding (OCE)**: Bottleneck fusion module ($512$ channels) integrating multi-scale teacher representations.
  - **Student Decoder**: Trainable symmetrical decoder with hierarchical reverse upsampling.
  - **Training Protocol**: Normal-only AutoVI images.
  - **Objective**: Multi-scale Cosine Feature Discrepancy:
    $$\mathcal{L}_{\text{RD}} = \sum_{k=1}^3 \left( 1 - \frac{1}{H_k W_k} \sum_{h, w} \frac{F_T^k(h, w) \cdot F_S^k(h, w)}{\|F_T^k(h, w)\|_2 \|F_S^k(h, w)\|_2} \right)$$
  - **Inference**: Multi-scale cosine distance anomaly map bilinearly interpolated to $(H, W)$ with Top-$K$ pixel mean anomaly scoring.
  - **Scientific Hypothesis**: Frozen ImageNet features provide rich semantics, while the trained student learns to reconstruct *only nominal industrial geometry*. Anomalous mechanical structures fail reconstruction, widening the anomaly distance.

### 3.2 Candidate B — Synthetic Anomaly Learning (DRAEM-style)
- **File**: `src/models/draem.py`
- **Theory (Zavrtanik et al., ICCV 2021)**:
  - **Reconstructive Subnetwork ($G_R$)**: Autoencoder / U-Net taking corrupted input $I_a$ and learning to restore nominal image $\hat{I}_r$.
    $$\mathcal{L}_{\text{rec}} = \mathcal{L}_2(I, \hat{I}_r) + 2 \cdot \mathcal{L}_{\text{SSIM}}(I, \hat{I}_r)$$
  - **Discriminative Subnetwork ($G_D$)**: 6-channel U-Net taking $[I_a, \hat{I}_r]$ and outputting pixel-level anomaly segmentation map $\hat{M}$.
    $$\mathcal{L}_{\text{disc}} = \mathcal{L}_{\text{Focal}}(\hat{M}, M) + \mathcal{L}_{\text{Dice}}(\hat{M}, M)$$
  - **Synthetic Anomaly Generator** (`src/dataset/synthetic_anomalies.py`):
    - Procedural Perlin noise masks & random polygon masks.
    - Donor cut-and-paste from other normal training images (simulating foreign objects / mislocated clips).
    - Geometric displacement (simulating loose retaining clips and displaced wiring).
    - Missing-region inpainting / deletion (simulating missing screws or clips).
    - Controlled texture noise and specular reflection simulation.
    - **Strict Boundary**: Zero test defect examples or test masks are ever accessed.

### 3.3 Baseline C — Existing Frozen PatchCore
- Retained as a frozen reference baseline with $6\%$ coreset sampling.

---

## 4. Local Engineering & Unit Test Verification (Gate A)

All implementations have undergone rigorous offline CPU verification. **All 37 unit and regression tests in the repository pass cleanly**:

```text
Ran 37 tests in 21.74s -> OK
```

Verified capabilities:
1. **Gradient & Weight Update Verification**: Proved student parameters compute non-zero gradients and update weights under AdamW, while teacher parameters remain strictly frozen.
2. **Loss Monotonicity**: Proved feature discrepancy loss decreases monotonically across training steps on nominal inputs.
3. **Synthetic Anomaly Engine**: Proved on-the-fly procedural injection of masks, donor patches, geometric shifts, and missing-region deletions without crashing or shape mismatches.
4. **Serialization & Reloading Fidelity**: Proved that saved checkpoints load standalone on CPU and reproduce numerical predictions with error $< 10^{-5}$.
5. **No Data Leakage**: Enforced that training dataloaders only consume `train/good` paths from canonical split manifests.

---

## 5. AWS SageMaker Adaptation & Infrastructure Specification

- **AWS Region**: `us-east-1`
- **AWS CLI Profile**: `autovi`
- **IAM Execution Role**: `arn:aws:iam::077578771161:role/service-role/AmazonSageMaker-ExecutionRole-20261115T084659`
- **Instance Type**: `ml.g4dn.xlarge` (1x NVIDIA Tesla T4 GPU, 16 GiB VRAM, 4 vCPUs, 16 GiB RAM)
- **Container Image**: `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker`
- **Training Entry Point**: `aws/sagemaker_train_deep.py`
- **S3 Data Prefix**: `s3://sagemaker-us-east-1-077578771161/autovi/`
- **S3 Phase 3 Output Prefix**: `s3://sagemaker-us-east-1-077578771161/autovi/phase3_outputs/`
- **Concurrency Quota**: Exactly 1 `ml.g4dn.xlarge` instance (enforced via active job query).

---

## 6. Pilot Experiment Plan & Cost Proposal

### 6.1 Proposed Single Pilot Experiment (Recommended First Step)
Before running full GPU training across all categories, we propose launching **one bounded pilot job**:
- **Target**: `pipe_clip`
- **Architecture**: **Reverse Distillation** (WideResNet-50-2 Teacher + Trainable Student Decoder)
- **Dataset**: 175 normal train images, 20 normal validation images
- **Hyperparameters**: 30 epochs, batch size 8, AdamW ($lr=2\times 10^{-4}$), mixed precision (AMP)
- **Expected Runtime**: $\sim 8$ minutes
- **Estimated Cost**: $\mathbf{\approx \$0.098\text{ USD}}$ ($\approx 10\text{ cents}$)

### 6.2 Full Phase 3 Experimental Budget (If Pilot is Approved & Successful)

| Job | Category | Model Type | Epochs | Batch | Estimated GPU Time | Estimated Cost (USD) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Pilot 1** | `pipe_clip` | Reverse Distillation | 30 | 8 | $\sim 8$ min | $\approx \$0.098$ |
| **Exp 2** | `pipe_clip` | DRAEM | 30 | 8 | $\sim 12$ min | $\approx \$0.147$ |
| **Exp 3** | `tank_screw` | Reverse Distillation | 30 | 8 | $\sim 10$ min | $\approx \$0.123$ |
| **Exp 4** | `tank_screw` | DRAEM | 30 | 8 | $\sim 14$ min | $\approx \$0.172$ |
| **Exp 5** | `engine_wiring` | Reverse Distillation | 30 | 8 | $\sim 10$ min | $\approx \$0.123$ |
| **Exp 6** | `engine_wiring` | DRAEM | 30 | 8 | $\sim 14$ min | $\approx \$0.172$ |
| **TOTAL (6 Jobs)**| — | — | — | — | **$\sim 68$ min** | **$\approx \$0.835\text{ USD}$** |

**Safety Cutoff**: Each job is configured with a strict 3,600-second (1-hour) timeout to prevent runaway billing.

---

## 7. Submission Commands Ready for Execution (Pending User Approval)

### Dry-Run Verification (Completed — Zero Cost)
```powershell
python autovi_project/aws/submit_deep_training.py --category pipe_clip --model-type reverse_distillation --dry-run
```

### Live Execution Command (Requires User Approval):
```powershell
python autovi_project/aws/submit_deep_training.py --category pipe_clip --model-type reverse_distillation --execute
```
