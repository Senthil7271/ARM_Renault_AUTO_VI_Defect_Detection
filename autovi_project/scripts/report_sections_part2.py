"""AutoVI IEEE 25-Page Report: Part 2 (Pages 9 to 16)"""

from pathlib import Path
from reportlab.platypus import Paragraph, Spacer, FrameBreak, PageBreak
from reportlab.lib import colors
from report_sections_part1_base import (
    STYLES, p, eq, h1, h2, h3, make_table, make_fig,
    FIG_DIR, RENDER_DIR, EXP_DIR
)

def get_page_9():
    """Page 9: Section IV-J (Threshold Calibration), Section V (AWS Cloud Training)."""
    col1 = [
        h2("J. Unsupervised Threshold Calibration & Statistical Limits"),
        p("In factory deployment, an anomaly detector must operate with a deterministic decision threshold tau: an image is flagged as anomalous if S_img >= tau, else passed as nominal. In strictly unsupervised settings, tau must be calibrated exclusively from normal validation data, without exposing the system to defect samples."),
        p("Our implementation (<code>src/training/trainer.py</code>) calibrates tau to the <b>95th percentile (target FPR = 0.05)</b> of nominal validation anomaly scores using a 90/10 split of normal training data:"),
        eq("\\tau_{95} = \\text{Quantile}\\left( \\{ S(I_{\\text{val}}^{(i)}) \\}_{i=1}^{N_{\\text{val}}}, \\; 0.95 \\right)"),
        p("<i>Statistical Sample Size Constraints:</i> Our formal audit (<code>reports/operational_confusion_matrices.md</code>) revealed that the 10% holdout split yields very small validation pools: <b>N_val = 17 to 38 images per category</b> (e.g., 20 images for <i>pipe_clip</i>, 17 for <i>underbody_pipes</i>). With N_val = 20, the empirical quantile step size is 1 / 20 = 5.0%, meaning the 95th percentile corresponds exactly to the second highest validation score."),
        p("Consequently, claiming strict theoretical false alarm bounds (e.g., <=0.5% or <=5.0%) on test data is mathematically unsupported with small sample sizes. Due to domain shift, minor lighting shifts, and unmodeled parts variance, empirical test false alarm rates range from <b>3.1%</b> (<i>underbody_pipes</i>) to <b>42.6%</b> (<i>pipe_staple</i>). As established in our industrial recommendations, robust factory gating requires scaling validation pools to N_val >= 200 normal samples."),
    ]
    col2 = [
        h1("V. Cloud Training on AWS SageMaker"),
        h2("A. High-Performance Cloud Training Architecture"),
        p("While final inference is executed on embedded edge NPUs, model training, feature extraction, and greedy coreset selection require high-throughput cloud infrastructure. We orchestrated our training pipeline on <b>Amazon Web Services (AWS) SageMaker</b>."),
        p("We provisioned SageMaker <code>ml.g4dn.xlarge</code> compute instances equipped with an <b>NVIDIA Tesla T4 GPU</b> (16 GB GDDR6 VRAM, 320 Turing Tensor Cores), 4 vCPUs (Intel Cascade Lake), and 16 GB of system RAM. Datasets and artifacts were hosted within a high-throughput Amazon S3 storage architecture (<code>s3://autovi-challenge-artifacts/</code>)."),
        p("Training jobs were submitted via a custom production automation harness (<code>autovi_project/aws/submit_deep_training.py</code>). The harness automatically validates IAM role permissions, provisions isolated Elastic Container Registry (ECR) PyTorch 2.1 containers, synchronizes training splits, and applies strict quota and cost safeguards."),
        p("Executing feature extraction on NVIDIA Tesla T4 GPUs accelerated multi-scale convolutional forward passes across thousands of high-resolution images, reducing training times from several hours on multi-core CPUs down to <b>under 4 minutes per category</b> on GPU."),
    ]
    return None, col1, col2

def get_page_10():
    """Page 10: Section V-B (SageMaker Pipeline), Section V-C (Distributed S3 Sync), Table IV."""
    col1 = [
        h2("B. Automated Source Packaging & Safeguards"),
        p("To guarantee reproducible cloud execution, <code>submit_deep_training.py</code> automatically builds a cryptographically signed source tarball (<code>deep_training_source.tar.gz</code>). The packager traverses the project root, aggregating entry scripts (<code>sagemaker_train_deep.py</code>), model modules, configuration YAMLs, and validated split manifests:"),
        eq("\\text{Hash}_{\\text{tarball}} = \\text{SHA-256}\\left( \\text{deep\\_training\\_source.tar.gz} \\right)"),
        p("<i>Production Safeguards:</i> Cloud training scripts frequently incur runaway costs due to unmonitored GPU instances. We integrated five critical safeguards:"),
        p("1) <b>Dry-Run Mode:</b> The default execution mode verifies all local paths, checks S3 connectivity, and prints execution parameters without triggering AWS API calls or billable charges."),
        p("2) <b>Max Concurrency Limits:</b> The harness queries active SageMaker jobs (<code>list_training_jobs</code>), refusing execution if concurrent jobs exceed quota limits (default limit: 2)."),
        p("3) <b>Max Runtime Hard-Cap:</b> Every training job enforces a strict <code>max_run</code> timeout of 3,600 seconds (1 hour), terminating orphaned jobs automatically."),
        p("4) <b>Real-Time Job Telemetry:</b> A background monitoring thread streams secondary status logs (Downloading -> Training -> Uploading -> Completed) at 30-second intervals."),
    ]
    col2 = [
        h2("C. Distributed Feature Extraction & S3 Synchronization"),
        p("Within the SageMaker training container, the training script loads pre-computed normal-only split manifests, constructs the WideResNet-50-2 feature extractor, and passes training images through GPU-accelerated forward hooks. Patch vectors from Layer 2 and Layer 3 are concatenated, normalized, and stored in GPU VRAM."),
        p("Greedy k-Center subsampling is parallelized using PyTorch CUDA tensor operations, performing distance updates directly in GPU memory to select optimal coreset subsets within seconds. Following coreset selection, validation samples are scored to calibrate tau_95, and all model artifacts (<code>patchcore_model.pkl</code>, <code>thresholds.json</code>, and evaluation curves) are serialized, compressed into <code>model.tar.gz</code>, and synchronized back to Amazon S3."),
        Spacer(1, 2),
    ]
    t4_hdr = ["Category", "Instance Type", "GPU / VRAM", "Feature Extract", "Coreset Time", "Total Job Time", "AWS Cost ($)"]
    t4_rows = [
        ["pipe_clip", "ml.g4dn.xlarge", "Tesla T4 (16GB)", "42.1 s", "18.4 s", "2 min 14 s", "$0.027"],
        ["pipe_staple", "ml.g4dn.xlarge", "Tesla T4 (16GB)", "39.8 s", "17.6 s", "2 min 08 s", "$0.026"],
        ["engine_wiring", "ml.g4dn.xlarge", "Tesla T4 (16GB)", "64.5 s", "31.2 s", "2 min 58 s", "$0.036"],
        ["tank_screw", "ml.g4dn.xlarge", "Tesla T4 (16GB)", "78.2 s", "38.5 s", "3 min 22 s", "$0.041"],
        ["underbody_pipes", "ml.g4dn.xlarge", "Tesla T4 (16GB)", "34.6 s", "14.2 s", "1 min 54 s", "$0.023"],
        ["underbody_screw", "ml.g4dn.xlarge", "Tesla T4 (16GB)", "84.1 s", "44.8 s", "3 min 41 s", "$0.044"],
        ["PORTFOLIO TOTAL", "ml.g4dn.xlarge", "Tesla T4 (16GB)", "343.3 s", "164.7 s", "16 min 17 s", "$0.197"],
    ]
    t4_elems = make_table(t4_hdr, t4_rows, [52, 42, 44, 34, 34, 36, 26], caption="IV: AWS SAGEMAKER TRAINING RESOURCE UTILIZATION & COSTS")
    col2.extend(t4_elems)
    return None, col1, col2

def get_page_11():
    """Page 11: Section V-D (Architecture Ablations on AWS), Section V-E (Frozen Models)."""
    col1 = [
        h2("D. Ablation of Alternative Deep Anomaly Models on AWS"),
        p("During Phase 3 research on AWS SageMaker, we conducted extensive pilot evaluations of competing deep anomaly detection architectures to determine whether generative or distillation approaches could surpass memory-retrieval PatchCore."),
        p("<i>1) Reverse Distillation (RD) Pilot:</i> We trained a teacher-student Reverse Distillation network on <i>pipe_clip</i>. While RD achieved acceptable performance on smooth surface scratches (Pixel AUROC 91.2%), it failed catastrophically on structural unlatched clips (Image AUROC <b>71.4%</b>, a -17.8% drop vs PatchCore). Cosine distance between student and teacher representations was dominated by high-frequency background sheet metal texture, while the subtle 3 mm latch displacement produced near-zero student divergence."),
        p("<i>2) DRAEM Inpainting Pilot:</i> We trained DRAEM using Perlin noise artifact synthesis. DRAEM achieved an Image AUROC of only <b>64.8%</b> on <i>pipe_clip</i>. Because Perlin noise simulates discoloration, blurring, and surface staining, the UNet reconstructor learned to inpaint texture patches but retained zero geometric understanding of rigid mechanical clips. When evaluated on test images where clips were cleanly missing, the autoencoder reconstructed the empty mounting hole perfectly, producing zero residual anomaly signal."),
    ]
    col2 = [
        h2("E. Frozen Models Policy & Macro Mathematical Budget"),
        p("Following our Phase 3 empirical ablations, we formally established that memory-retrieval PatchCore represents the superior architectural paradigm for automotive mechanical inspection. To preserve benchmark stability and prevent regressions on categories already operating at world-class performance, we instituted a <b>Frozen Models Policy</b>:"),
        p("Models for <i>underbody_pipes</i> (0.9980 AUROC), <i>underbody_screw</i> (0.9886 AUROC), and <i>pipe_staple</i> (0.9534 AUROC) were locked with immutable cryptographic SHA-256 hashes (persisted in <code>reports/frozen_models_registry.json</code>)."),
        p("<i>Macro Mathematical Budget:</i> To achieve our master research mandate of Six-Category Macro AUROC >= 0.9500 (5.7000 sum total), and given the frozen subset sum of 2.9400, the remaining three underperforming categories (<i>tank_screw</i>, <i>pipe_clip</i>, <i>engine_wiring</i>) required a combined sum of >= 2.7600 (Mean >= 0.9200):"),
        eq("\\text{Target: } \\frac{2.9400 + 0.9000 + 0.9500 + 0.9200}{6} = \\mathbf{0.9517} \\ge 95.0\\%"),
        p("Through Config C (ROI + Coordinate Conditioning), our final models far exceeded this target, reaching 0.9998 (<i>tank_screw</i>), 0.9432 (<i>pipe_clip</i>), and 0.9691 (<i>engine_wiring</i>), establishing a record portfolio Macro AUROC of <b>97.53%</b>."),
    ]
    return None, col1, col2

def get_page_12():
    """Page 12: Section VI (Hailo-8 NPU Hardware Compilation), VI-A, VI-B."""
    col1 = [
        h1("VI. Hardware Compilation & Quantization for Hailo-8 NPU"),
        h2("A. Hailo-8 AI Accelerator Architecture"),
        p("Deploying our 68.9-million-parameter WideResNet-50-2 backbone onto the Raspberry Pi 5 QC station requires dedicated hardware acceleration. We target the <b>Hailo-8 M.2 AI Acceleration Module</b> (mounted via the official Raspberry Pi AI HAT+)."),
        p("Unlike conventional GPUs and VPUs that rely upon standard Von Neumann memory hierarchies (shuffling weights and intermediate activations back and forth across power-hungry external DRAM buses), the Hailo-8 utilizes a revolutionary <b>Structure-Defined Dataflow Architecture</b> [11]."),
        p("The Hailo-8 compute core consists of a highly scalable 2D grid of heterogeneous compute and memory units. Neural network layers are spatially mapped directly onto the physical hardware fabric: weights are stored in localized on-chip SRAM cells situated adjacent to multiply-accumulate (MAC) units, and feature tensors stream directly from one layer's hardware blocks into subsequent layers through dedicated on-chip interconnections."),
        p("This dataflow paradigm delivers <b>26 Tera-Operations Per Second (TOPS)</b> of compute capacity at an ultra-low power consumption profile of <b>under 2.5 Watts</b>, achieving an unprecedented compute efficiency (>10 TOPS/Watt) that is ideally suited for sealed automotive inspection stations."),
    ]
    col2 = [
        h2("B. Hailo Dataflow Compiler (DFC) Pipeline"),
        p("To compile our WideResNet-50-2 feature extractor onto the Hailo-8 fabric, we implemented an automated compilation toolchain leveraging the <b>Hailo Dataflow Compiler (DFC) v3.28</b>."),
        p("The compilation pipeline proceeds across four sequential phases:"),
        p("1) <b>ONNX Translation:</b> The PyTorch model is exported to an optimized ONNX computational graph (<code>wideresnet50_patchcore.onnx</code>, 99.5 MB) with static input dimensions [1, 3, 256, 256]. The DFC parses the ONNX graph into a Hailo Internal Representation Archive (HAR)."),
        p("2) <b>Model Script Configuration:</b> Using a custom model script (<code>wideresnet50_patchcore.alls</code>), we configure on-chip preprocessing: the Hailo-8 hardware automatically receives raw UINT8 camera buffers [0, 255], normalizes them using ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225], and casts them to internal tensor formats without consuming host CPU cycles."),
        p("3) <b>Multi-Scale Output Tapping:</b> We tap intermediate activation outputs from Layer 2 (Conv3_x) and Layer 3 (Conv4_x), configuring on-chip bilinear upsampling to align spatial dimensions to 32 x 32."),
        p("4) <b>Post-Training Quantization:</b> We apply full INT8 quantization across all weights and intermediate activation tensors."),
    ]
    return None, col1, col2

def get_page_13():
    """Page 13: Section VI-C (INT8 PTQ Formulation), Section VI-D (DFC Optimization Passes)."""
    col1 = [
        h2("C. INT8 Post-Training Quantization Formulation"),
        p("Deep neural networks trained in 32-bit floating-point (FP32) require massive memory bandwidth and complex arithmetic logic units. Post-Training Quantization (PTQ) maps continuous 32-bit floating-point weights and activations into 8-bit signed integers (INT8, range [-128, 127]), achieving a 4x reduction in model footprint and enabling ultra-fast integer arithmetic."),
        p("For convolutional weights W in R, we employ <b>symmetric per-channel quantization</b>:"),
        eq("W_{\\text{int8}} = \\text{clip}\\left( \\left\\lfloor \\frac{W}{S_w} \\right\\rceil, \\; -128, \\; 127 \\right), \\quad S_w = \\frac{\\max(|W|)}{127}"),
        p("For activation feature maps X in R, we employ <b>asymmetric per-tensor quantization</b> to accommodate asymmetric distributions resulting from ReLU activations:"),
        eq("X_{\\text{int8}} = \\text{clip}\\left( \\left\\lfloor \\frac{X}{S_x} \\right\\rceil + Z_x, \\; -128, \\; 127 \\right)"),
        p("where S_x is the activation scale factor and Z_x is the integer zero-point offset."),
        p("<i>Zero Defect Exposure Calibration:</i> Unlike supervised quantization, PTQ for unsupervised anomaly detection must preserve nominal feature manifold geometry without ever observing defect samples. We constructed a <b>150-image calibration dataset</b> uniformly sampled across nominal training images (25 images per category from <code>train/good/</code>). Zero defective images were included during calibration."),
    ]
    col2 = [
        h2("D. DFC Optimization & Allocation Passes"),
        p("During Post-Training Quantization, the Hailo Dataflow Compiler executes automated graph optimization algorithms to minimize quantization noise:"),
        p("1) <b>Layer Fusion:</b> Consecutive operations (Convolution + Batch Normalization + ReLU activation) are mathematically folded into unified integer execution layers, eliminating intermediate DRAM memory round-trips."),
        p("2) <b>Cross-Layer Equalization:</b> Activations in deep networks frequently exhibit wide dynamic range disparities across adjacent channels. The DFC applies weight balancing transformations that rescale weight channels to equalize activation ranges without altering mathematical outputs."),
        p("3) <b>Gradient-Free Calibration:</b> Using the 150 normal calibration images, the DFC computes Kullback-Leibler (KL) divergence between FP32 activation histograms and INT8 quantized distributions, selecting optimal clipping thresholds that preserve feature density."),
        p("4) <b>Resource Allocation:</b> The compiler maps the quantized computational graph onto the physical Hailo-8 compute clusters, synthesizing routing paths that maximize parallel throughput. The final output is serialized into a standalone <b>23.4 MB Hailo Executable Format (HEF) binary</b> (<code>wideresnet50_patchcore.hef</code>)."),
    ]
    return None, col1, col2

def get_page_14():
    """Page 14: Section VI-E (Numerical Sensitivity), Section VI-F (Parity & HEF), Fig 4."""
    col1 = [
        h2("E. INT8 Numerical Sensitivity & Cosine Stability"),
        p("A critical scientific question in edge deployment is whether INT8 quantization introduces numerical noise that alters nearest-neighbor patch rankings in the coreset memory bank."),
        p("We conducted an exhaustive numerical parity analysis (<code>reports/hailo_quantization_parity.md</code>), comparing FP32 PyTorch embeddings against INT8 quantized representations across all 2,399 AutoVI test images."),
        p("<i>High-Dimensional Cosine Stability in R^1536:</i> Our WideResNet-50-2 patch embeddings inhabit a high-dimensional Euclidean space (D = 1,536). In 1536-dimensional space, random isotropic INT8 truncation noise epsilon is statistically orthogonal to the underlying semantic feature vector phi with probability approaching 1:"),
        eq("\\lim_{D \\to \\infty} \\langle \\phi, \\epsilon \\rangle = 0"),
        p("Consequently, the directional cosine similarity between FP32 and INT8 patch embeddings remains extraordinarily stable. Across all evaluated test patches, the <b>minimum observed cosine similarity was 0.999981</b>, with a mean feature Mean Absolute Error (MAE) of only <b>0.003958</b>."),
        p("Because PatchCore anomaly scoring is governed by relative topological distances rather than absolute vector scale, top-1 nearest-neighbor coreset matches were preserved across <b>>99.4% of evaluated patches</b>."),
    ]
    col2 = [
        h2("F. Benchmark Parity & Physical HEF Compilation"),
        p("To verify benchmark integrity, we evaluated the INT8 quantized backbone under the official AutoVI benchmark protocol. As summarized in Table IX, the macro-average Image AUROC dropped by an imperceptible <b>Delta = -0.0002</b> (97.53% FP32 vs 97.51% INT8), while official sPRO@0.30 dropped by only <b>Delta = -0.0004</b> (97.00% FP32 vs 96.96% INT8)."),
        p("Calibrated operational thresholds tau_95 required zero recalibration under Hailo INT8 execution, confirming that the Hailo-8 NPU delivers full floating-point accuracy at edge hardware speeds."),
        Spacer(1, 2),
    ]
    fig4 = make_fig(FIG_DIR / "fig4_quantization_parity.png", 238, 100, 4, "INT8 Post-Training Quantization parity: layer-wise cosine similarity (>= 0.999981) and reconstruction error (MAE = 0.003958).")
    col2.extend(fig4)
    return None, col1, col2

def get_page_15():
    """Page 15: Section VII (Raspberry Pi 5 Edge Deployment), VII-A, VII-B."""
    col1 = [
        h1("VII. Embedded Edge Deployment on Raspberry Pi 5"),
        h2("A. System Hardware Architecture"),
        p("To validate real-world production viability, we deployed our complete inspection pipeline onto an industrial edge quality inspection station built upon the <b>Raspberry Pi 5 single-board computer</b> paired with the <b>Raspberry Pi AI HAT+</b>."),
        p("<i>1) Compute Platform:</i> The Raspberry Pi 5 features the Broadcom BCM2712 application processor, integrating a <b>quad-core Arm Cortex-A76 CPU cluster operating at 2.4 GHz</b> with 512 KB L2 cache per core and 2 MB shared L3 cache, supported by <b>16 GB of high-speed LPDDR4X SDRAM</b> (4267 MT/s)."),
        p("<i>2) Hardware AI Acceleration:</i> The Raspberry Pi AI HAT+ interfaces with the Raspberry Pi 5 via a dedicated <b>PCIe 2.0 x1 interface</b> running at 5.0 GT/s, providing a peak bidirectional interconnect bandwidth of <b>500 MB/s</b>. The AI HAT+ hosts the Hailo-8 M.2 module, delivering 26 TOPS of dedicated NPU compute."),
        p("<i>3) Thermal Management:</i> Modern automotive factories present harsh thermal environments (ambient temperatures reaching 45 deg C). We integrated the Raspberry Pi Active Cooler (combining an aluminum heatsink with a PWM-controlled blower fan). Under continuous 100 FPS inspection workloads, system temperatures stabilized at <b>56.2 deg C</b>, safely below the 85 deg C thermal throttling threshold."),
    ]
    col2 = [
        h2("B. Operating System & Kernel Driver Stack"),
        p("The edge station runs <b>Raspberry Pi OS (Bookworm 64-bit)</b> based on Debian 12, powered by Linux Kernel 6.6.y."),
        p("<i>1) PCIe Kernel Module:</i> Hardware communication with the Hailo-8 accelerator is managed via the open-source <code>hailo_pci</code> kernel driver. The driver configures Direct Memory Access (DMA) channels between host system RAM and the Hailo-8 M.2 module, enabling zero-copy frame transfers."),
        p("<i>2) HailoRT Runtime Library:</i> Model execution is orchestrated via <b>HailoRT v4.18</b> and its official Python bindings (<code>pyhailort</code>). HailoRT configures asynchronous Virtual Streams (VStreams) that pipeline camera frame acquisition, NPU inference, and output tensor transfer."),
        p("<i>3) Hardware Diagnostics:</i> We implemented an automated edge validation script (<code>tools/hailo_diagnostics.sh</code>) that verifies PCIe link negotiation, monitors on-chip junction temperatures, and executes a smoke test verifying tensor checksums against host golden reference outputs prior to enabling production inspection gates."),
    ]
    return None, col1, col2

def get_page_16():
    """Page 16: Section VII-C (Dual-Backend Inference Engine), Section VII-D (Memory Banks), Table V."""
    col1 = [
        h2("C. Dual-Backend Edge Inference Engine Architecture"),
        p("To guarantee absolute factory uptime, we architected the edge inference engine (<code>autovi-edge/src/inference_engine.py</code>) with a robust dual-backend failover design:"),
        p("1) <b>HailoInferenceBackend (Primary):</b> Leverages native <code>hailo_platform</code> VStreams. Raw UINT8 camera frames (256x256x3) are streamed directly to the Hailo-8 NPU over PCIe. The NPU executes in-hardware normalization, multiscale convolutional feature extraction, and bilinear pooling, returning concatenated 1536-dimensional patch representations in only <b>7.98 ms</b>."),
        p("2) <b>ONNXRuntimeBackend (Fallback):</b> If the PCIe accelerator is unseated or encounters hardware faults, the engine gracefully transitions to an Arm-optimized ONNX Runtime CPU engine. While latency increases to ~260 ms, inspection integrity is maintained without stopping the assembly line."),
        p("<i>Nearest-Neighbor Vectorization:</i> Following feature extraction, query patches are matched against the active category coreset bank. We vectorized Euclidean distance computation via matrix multiplication:"),
        eq("\\| Q - M \\|^2 = \\text{diag}(Q Q^T) \\mathbf{1}^T + \\mathbf{1} \\text{diag}(M M^T)^T - 2 Q M^T"),
        p("reducing NN search latency on the Arm Cortex-A76 to only <b>1.2 ms</b> for 8,960 coreset patches."),
    ]
    col2 = [
        h2("D. Dynamic Category Hot-Swapping"),
        p("Automotive mixed-model assembly lines frequently assemble multiple vehicle variants on a single track. Stopping the inspection server to reload model weights introduces unacceptable cycle delays."),
        p("Our edge inference engine pre-loads coreset memory banks for all six categories into system RAM during initialization (total RAM footprint: <b>429.98 MB</b>). An operator or PLC command can dynamically switch the active inspection category via an API call in <b>under 0.8 milliseconds</b> with zero process restart."),
        Spacer(1, 2),
    ]
    t5_hdr = ["Component Category", "Memory Patches", "Memory RAM", "Switch Latency", "Hailo NPU (ms)", "Total Edge (ms)"]
    t5_rows = [
        ["pipe_clip", "8,960", "55.05 MB", "0.78 ms", "7.98 ms", "9.18 ms"],
        ["pipe_staple", "8,755", "53.79 MB", "0.76 ms", "7.98 ms", "9.14 ms"],
        ["engine_wiring", "13,107", "80.53 MB", "0.82 ms", "7.98 ms", "9.42 ms"],
        ["tank_screw", "14,643", "89.97 MB", "0.85 ms", "7.98 ms", "9.56 ms"],
        ["underbody_pipes", "7,372", "45.29 MB", "0.72 ms", "7.98 ms", "8.98 ms"],
        ["underbody_screw", "17,152", "105.38 MB", "0.89 ms", "7.98 ms", "9.84 ms"],
        ["PORTFOLIO TOTAL", "69,989", "429.98 MB", "Mean: 0.80 ms", "7.98 ms", "Mean: 9.35 ms"],
    ]
    t5_elems = make_table(t5_hdr, t5_rows, [52, 40, 38, 36, 36, 36], caption="V: EDGE MEMORY BANK ALLOCATION & EXECUTION LATENCIES")
    col2.extend(t5_elems)
    return None, col1, col2

print("report_sections_part2.py ready.")
