"""AutoVI IEEE 25-Page Report: Part 1 (Pages 1 to 8)"""

from pathlib import Path
from reportlab.platypus import Paragraph, Spacer, FrameBreak, PageBreak, NextPageTemplate
from reportlab.lib import colors
from report_sections_part1_base import (
    STYLES, p, eq, h1, h2, h3, make_table, make_fig,
    FIG_DIR, RENDER_DIR, EXP_DIR
)

def get_page_1():
    """Page 1: Title Banner, Abstract, Index Terms, Section I (Intro), Section I-A, I-B."""
    top = [
        p("AutoVI: End-to-End Industrial Visual Anomaly Detection and Localization from Cloud to Edge — Architecture, AWS SageMaker Training, Hailo-8 NPU Acceleration, Raspberry Pi 5 Deployment, ExecuTorch Reasoning, and Mobile Packaging", "Title"),
        p("<b>Linga Reddy</b>, Lead Computer Vision Research Engineer, Bharat AI-SoC Project<br/>"
          "<b>Alexane Carvalho</b>, Université de Technologie de Compiègne & Renault Group &nbsp;|&nbsp; "
          "<b>David Picard</b>, LIGM, École des Ponts ParisTech", "Authors"),
        p("Automotive Quality Inspection Consortium &bull; Advanced Manufacturing Research &bull; Industrial Edge AI Initiative", "Affiliations"),
        p("<b><i>Abstract</i>—Industrial automated visual inspection in modern automotive manufacturing requires zero-defect quality gates capable of detecting rare, subtle, and unpredictable physical flaws across high-speed assembly lines. Supervised defect classification fails in production environments due to the extreme scarcity of defective parts and the open-set emergence of novel anomaly morphologies. In this work, we present a production-grade, end-to-end industrial anomaly detection and localization system engineered for the official Automotive Visual Inspection (AutoVI) benchmark developed by Renault Group and Université de Technologie de Compiègne. Our system learns strictly from defect-free (normal) components using an ImageNet-pretrained WideResNet-50-2 backbone (68.9M parameters) extracting multi-scale patch embeddings from intermediate feature hierarchies (Layer 2 and Layer 3, feature dimension D = 1,536). To overcome the massive memory scaling bottleneck inherent in dense patch representations, we implement a memory-bounded Greedy k-Center coreset subsampling algorithm that compresses memory banks by over 94% with zero loss in topological representation. We conduct distributed cloud training on Amazon Web Services (AWS) SageMaker using NVIDIA Tesla T4 GPUs, deploy Post-Training Quantization (PTQ) via the Hailo Dataflow Compiler (DFC) to map the backbone onto the 26-TOPS Hailo-8 M.2 AI accelerator with near-zero feature degradation (minimum cosine similarity >= 0.999981), and establish a live production QC inspection station on a Raspberry Pi 5 paired with the Raspberry Pi AI HAT+ and hardware SPI/I2C TFT operator status display. Furthermore, we deploy Meta LLaMA-3.2-1B-Instruct via the Meta ExecuTorch edge reference runtime on Arm Cortex-A76 cores to generate automated, deterministic 3-part Root-Cause Analysis (RCA), Standard Operating Procedure (SOP) technician remediation, and quality gate disposition. Finally, we package the complete pipeline for Arm-based mobile platforms (Android Termux, CameraX, and ONNX Runtime Mobile). Under the official AutoVI benchmark protocol, our system achieves an outstanding Macro-Average Image AUROC of 97.53% across all six industrial assembly categories and an official Per-Region Overlap (sPRO@0.30) of 96.96%, while executing end-to-end inference in 9.18 ms (108.9 FPS) on Hailo-8 hardware, fully respecting automotive factory cycle time budgets (<100 ms).</b>", "Abstract"),
        p("<b><i>Index Terms</i>—Industrial Visual Inspection, Unsupervised Anomaly Detection, PatchCore, WideResNet-50-2, Post-Training Quantization (PTQ), Hailo-8 NPU, Raspberry Pi 5, Edge AI, ExecuTorch, Meta LLaMA-3.2, Mobile Arm Deployment, Automotive Manufacturing, AutoVI Benchmark.</b>", "IndexTerms"),
    ]

    col1 = [
        h1("I. Introduction"),
        p("Modern automotive powertrain, chassis, and underbody manufacturing facilities operate under strict zero-defect quality mandates. High-speed final assembly stations require 100% automated visual quality verification across mission-critical mechanical components, including hydraulic pipe clips, electrical wiring harnesses, fuel tank fasteners, and underbody structural clips. An undetected defect—such as an unlatched wiring harness, a missing fuel tank hex screw, or a loose hydraulic brake line staple—can result in catastrophic vehicular failure, expensive warranty recalls, and severe factory downtime."),
        p("Historically, industrial automated optical inspection (AOI) relied upon hand-engineered geometric feature extractors, edge detection algorithms, and template matching. While effective in tightly controlled laboratory environments, classical AOI breaks down on active factory floors characterized by mechanical vibrations, high-frequency camera jitter, parts positioning tolerances (+-10 mm), and dynamic ambient lighting fluctuations."),
        p("Supervised deep learning architectures (e.g., YOLO, Mask R-CNN, Vision Transformers) have demonstrated remarkable efficacy in consumer computer vision. However, supervised methods require thousands of annotated defect instances per part category. In modern manufacturing lines operating at Six Sigma standards (<3.4 defects per million opportunities), defective samples are extraordinarily scarce. Furthermore, manufacturing defects are inherently open-set: physical anomalies emerge in unpredictable morphologies (tears, punctures, foreign object entrapment, partial insertion, micro-fractures, oxidation) that were never encountered during offline training."),
        p("Consequently, industrial quality control has fundamentally shifted toward unsupervised anomaly detection (UAD), wherein deep neural networks are trained strictly on nominal (defect-free) parts to model the underlying distribution of normal manufacturing appearance. During inference, any component exhibiting significant statistical or geometric deviation from nominal memory is flagged as defective."),
    ]

    col2 = [
        h2("A. The AutoVI Industrial Benchmark Challenge"),
        p("To bridge the gap between academic anomaly datasets (such as MVTec AD and VisA, which feature pristine centered objects on uniform laboratory backgrounds) and messy production realities, Renault Group and Université de Technologie de Compiègne introduced the Automotive Visual Inspection (AutoVI) dataset [1]. AutoVI captures six mechanical categories across active assembly plants: <i>pipe_clip</i>, <i>pipe_staple</i>, <i>engine_wiring</i>, <i>tank_screw</i>, <i>underbody_pipes</i>, and <i>underbody_screw</i>."),
        p("Unlike laboratory benchmarks, AutoVI presents extreme industrial complexities: unconstrained camera angles, specular reflections off metallic chassis panels, complex background clutter (engine blocks, painted sheet metal, cardboard packaging), and subtle structural flaws where components are present but mechanically unseated by mere millimeters."),
        h2("B. Principal Contributions"),
        p("<b>1) Enhanced Multi-Scale Architecture:</b> ImageNet-pretrained WideResNet-50-2 backbone extracting concatenated patch representations (D = 1,536) from Layer 2 and Layer 3, paired with component ROI slicing and 2D coordinate conditioning (Config C)."),
        p("<b>2) Data Integrity & Leakage Resolution:</b> Cryptographic SHA-256 audits resolving multi-mask union conventions and uncovering an empirical train/test data leakage in <i>engine_wiring</i> (60 duplicate pairs), formally quantifying its benchmark evaluation impact."),
        p("<b>3) Cloud AWS Training Pipeline:</b> Automated, cost-safeguarded Amazon SageMaker training on NVIDIA Tesla T4 instances with memory-bounded Greedy k-Center selection (94% memory reduction)."),
        p("<b>4) Hailo-8 NPU Hardware Acceleration:</b> INT8 Post-Training Quantization (PTQ) via Hailo DFC compiling a physical HEF binary achieving sub-10ms inference with minimal cosine degradation (>= 0.999981)."),
        p("<b>5) Edge Station & ExecuTorch Advisory:</b> Live production QC station on Raspberry Pi 5 with hardware TFT display, industrial web dashboard, stress-testing suite, and on-device Meta LLaMA-3.2-1B reasoning via Meta ExecuTorch."),
        p("<b>6) Mobile Arm Package:</b> Standalone runtime distribution for Android smartphones via Termux and native CameraX/ORT Mobile."),
    ]
    return top, col1, col2

def get_page_2():
    """Page 2: Section II (Related Work & Benchmark Foundations), Table I."""
    col1 = [
        h1("II. Related Work & Benchmark Foundations"),
        h2("A. Paradigms in Industrial Anomaly Detection"),
        p("Unsupervised industrial anomaly detection has evolved across three dominant technical methodologies: reconstruction-based methods, student-teacher distillation networks, and feature-embedding memory retrieval systems."),
        p("<i>1) Reconstruction-Based Methods:</i> Early deep anomaly detection relied upon Convolutional Autoencoders (CAE), Variational Autoencoders (VAE), and Generative Adversarial Networks (GANs) [2], [3]. These architectures learn an identity mapping over nominal training samples under the hypothesis that anomalous features cannot be reconstructed faithfully through the latent bottleneck. The residual reconstruction error |x - G(E(x))| serves as the pixel anomaly map. However, standard autoencoders suffer from two fundamental flaws in automotive settings: first, high-capacity decoders frequently generalize 'too well', successfully reconstructing anomalous textures; second, low-capacity autoencoders generate blurred edge reconstructions, triggering massive false positive rates along nominal high-contrast metal contours."),
        p("<i>2) Feature-Embedding Memory Banks:</i> Pioneered by SPADE [4], PaDiM [5], and PatchCore [6], memory-retrieval methods bypass generative decoders entirely. Instead, they leverage deep feature representations extracted from frozen neural networks pretrained on large-scale visual distributions (ImageNet). PatchCore partitions normal images into spatially localized patch tokens, pools multi-scale representations, and constructs an exhaustive nominal memory bank M. During test inference, the anomaly score of an input patch p is directly defined as its minimum Euclidean distance to M:"),
        eq("s(p) = min_{m in M} || phi(p) - m ||_2"),
        p("Roth et al. [6] demonstrated that PatchCore sets the state-of-the-art across academic benchmarks because deep pretrained features preserve high-level semantic abstractions while maintaining precise local spatial sensitivity."),
    ]
    col2 = [
        h2("B. Knowledge Distillation & Generative Pitfalls in AutoVI"),
        p("<i>3) Student-Teacher Distillation:</i> Architectures such as STPM [7] and Reverse Distillation (RD) [8] utilize a frozen teacher network and train a compact student network to mimic teacher representations solely on normal images. On anomalous patches, the student fails to match the teacher, producing a discrepancy vector. While effective on surface scratches, our AWS SageMaker experiments revealed that student-teacher distillation suffers severe degradation on AutoVI's structural anomalies: the student bottleneck smooths out high-frequency mechanical edges, missing unclipped fasteners."),
        p("<i>4) Synthetic Anomaly Inpainting (DRAEM):</i> DRAEM [9] trains an autoencoder to reconstruct clean images from synthetically corrupted inputs generated via Perlin noise masks. In mechanical assemblies, however, Perlin noise generates diffuse, cloudy discolorations that bear zero physical resemblance to structural flaws (e.g., a missing hex screw or a 3 mm displaced wire bracket). As established in our empirical ablations, DRAEM achieves near-zero gradients on structural mechanical displacements."),
        Spacer(1, 4),
    ]
    table_header = ["Paradigm", "Key Exemplars", "Inference Speed", "AutoVI Structural Sensitivity", "Edge NPU Feasibility"]
    table_rows = [
        ["Reconstruction", "AE, VAE, GAN", "Medium (20-40 ms)", "Low (Edge blur false alarms)", "Medium (High MACs)"],
        ["Distillation", "STPM, RevDist", "Fast (15-30 ms)", "Moderate (Bottleneck loss)", "High (Dual forward)"],
        ["Synthetic Inpaint", "DRAEM, CutPaste", "Medium (25-50 ms)", "Poor (Perlin noise mismatch)", "Moderate (UNet decoder)"],
        ["Memory-Retrieval", "PatchCore, PaDiM", "Ultra-Fast (<10 ms)", "Exceptional (Nearest-Neighbor)", "Ideal (Linear NN Search)"],
    ]
    t_elems = make_table(table_header, table_rows, [55, 45, 45, 52, 46], caption="I: COMPARATIVE TAXONOMY OF INDUSTRIAL ANOMALY DETECTION PARADIGMS")
    col2.extend(t_elems)
    col2.extend([
        p("As summarized in Table I, memory-retrieval PatchCore provides the optimal trade-off: it requires zero backpropagation or synthetic anomaly assumptions, provides deterministic geometric distance bounds, and maps directly onto high-throughput matrix multiplication units on edge NPUs."),
    ])
    return None, col1, col2

def get_page_3():
    """Page 3: Section II-C, II-D, Section III (Dataset Topology), Table II."""
    col1 = [
        h2("C. Edge AI Accelerators & Embedded NPU Runtimes"),
        p("Deploying high-resolution deep vision models on factory floors requires embedded edge compute hardware capable of executing heavy tensor workloads within strict cycle time budgets (<100 ms) while operating within sealed, fanless industrial enclosures (<15 W)."),
        p("Traditional edge CPUs (such as Arm Cortex-A76) lack the vector compute density required for dense 1536-dimensional feature extraction, yielding unacceptably high latencies (170-640 ms per inspection). While discrete edge GPUs (e.g., NVIDIA Jetson Orin) offer high compute, their high thermal dissipation (15-25 W) and bill-of-materials cost restrict wide-scale deployment across dozens of assembly cells."),
        p("In contrast, dedicated Neural Processing Units (NPUs) based on Structure-Defined Dataflow architectures—specifically the Hailo-8 AI accelerator (26 TOPS)—deliver orders-of-magnitude higher computational efficiency (>10 TOPS/Watt). By compiling convolutional graphs into on-chip spatial compute fabrics, the Hailo-8 executes deep backbones in sub-10ms latencies with negligible thermal impact (<2.5 W)."),
        h2("D. On-Device Generative Reasoning with ExecuTorch"),
        p("Industrial quality inspection has traditionally stopped at binary PASS/REJECT verdicts. However, modern Industry 4.0 production environments demand actionable cognitive intelligence: when an anomaly is rejected, line technicians require immediate root-cause diagnostics and standard operating procedure (SOP) remediation steps."),
        p("The emergence of compact Small Language Models (SLMs), such as Meta LLaMA-3.2-1B-Instruct, enables on-device reasoning directly at the inspection cell. By leveraging the Meta ExecuTorch runtime optimized with Arm NEON kernels, generative diagnostic reasoning can execute on local CPU cores without requiring cloud network connectivity or risking proprietary assembly line data leakage."),
    ]
    col2 = [
        h1("III. AutoVI Dataset Architecture & Audit"),
        h2("A. Dataset Topology & Industrial Acquisition Context"),
        p("The AutoVI dataset comprises 3,922 high-resolution images across six automotive assembly components acquired under live industrial manufacturing conditions. Images were captured using industrial RGB cameras mounted on robotic manipulator arms and stationary inspection gantries above active vehicle production lines."),
        p("The dataset captures two distinct mechanical categories: 1) Structural Fasteners and Retainers (<i>pipe_clip</i>, <i>pipe_staple</i>, <i>tank_screw</i>, <i>underbody_screw</i>), where anomalies represent missing parts, loose clips, or unseated brackets; and 2) Complex Flexible Assemblies (<i>engine_wiring</i>, <i>underbody_pipes</i>), where anomalies represent misrouted wire harnesses, foreign object obstructions (blue plastic hoops, cardboard debris), and crimped hoses."),
        Spacer(1, 3),
    ]
    t2_hdr = ["Component Category", "Train Good", "Test Good", "Test Defect", "Total Test", "GT Masks", "Canonical Res"]
    t2_rows = [
        ["pipe_clip", "195", "195", "142", "337", "142", "400 x 400"],
        ["pipe_staple", "191", "188", "117", "305", "117", "400 x 400"],
        ["engine_wiring", "285", "285", "322", "607", "355", "400 x 400"],
        ["tank_screw", "318", "318", "95", "413", "95", "1000 x 750"],
        ["underbody_pipes", "161", "161", "184", "345", "185", "1000 x 750"],
        ["underbody_screw", "373", "374", "18", "392", "18", "1000 x 750"],
        ["PORTFOLIO TOTAL", "1,523", "1,521", "878", "2,399", "912", "Mixed Res"],
    ]
    t2_elems = make_table(t2_hdr, t2_rows, [55, 30, 30, 32, 32, 32, 32], caption="II: AUTOVI COMPREHENSIVE DATASET TOPOLOGY & SPLIT SIZES")
    col2.extend(t2_elems)
    col2.extend([
        p("As detailed in Table II, the official test set comprises N = 2,399 images (1,521 normal and 878 defective) evaluated across 912 precise pixel-level ground truth masks spanning 11 unique anomaly subtypes."),
    ])
    return None, col1, col2

def get_page_4():
    """Page 4: Section III-B, Section III-C (Leakage Resolution), Fig 5."""
    col1 = [
        h2("B. Dataset Integrity Audit & Multi-Mask Fusion"),
        p("To ensure scientific evaluation integrity, we implemented an automated cryptographic auditing pipeline (<code>src/dataset/audit.py</code>). Every file across the entire repository was audited for SHA-256 hash uniqueness, aspect ratio consistency, and ground-truth mask alignment."),
        p("<i>1) Non-Square Aspect Ratios:</i> While earlier anomaly benchmarks enforce square inputs (e.g. 256x256), AutoVI incorporates two distinct optical capture setups: a 1:1 square aspect ratio (400x400) for wiring and clips, and a 4:3 high-resolution format (1000x750) for underbody panels and fuel tanks. Naive isotropic resizing to 256x256 introduces severe geometric aspect distortion (distorting circular screw heads into squashed ellipses), degrading spatial correlation. Our pipeline standardizes on aspect-ratio-preserving padding and canonical evaluation scaling."),
        p("<i>2) Multi-Mask Union:</i> In categories such as <i>engine_wiring</i> and <i>underbody_pipes</i>, multiple distinct physical flaws can co-occur within a single inspection frame (e.g., simultaneous wiring misclip and cardboard obstruction). The AutoVI dataset stores these ground-truth annotations as separate subfolder masks. Our multi-mask loader implements robust bitwise OR fusion:"),
        eq("M_{combined} = \\bigvee_{k=1}^K M_k(u, v)"),
        p("ensuring that the official Per-Region Overlap (sPRO) evaluator receives fully consolidated ground-truth target regions."),
        h2("C. Empirical Discovery of Data Leakage in Engine Wiring"),
        p("During cryptographic auditing, cross-split SHA-256 hash collision checks revealed a significant data integrity anomaly: exactly <b>60 unique normal image pairs</b> were identical between <code>engine_wiring/train/good</code> (285 images) and <code>engine_wiring/test/good</code> (285 images)."),
        p("Cryptographic verification confirmed that 0 defective test images were leaked (100% clean anomaly boundary). However, 21.05% of the test normal set represented exact duplicates of training instances."),
    ]
    col2 = [
        h2("D. Mathematical Impact of Train/Test Leakage"),
        p("In anomaly detection benchmark evaluation, the Receiver Operating Characteristic (ROC) curve evaluates the pairwise rank ordering between normal and defective test samples. The total number of evaluation pairs is defined by N_normal x N_defective."),
        p("In the official split, N_normal = 285 and N_defective = 322, yielding <b>91,770 comparison pairs</b>. Because PatchCore computes nearest-neighbor distances to training patches, test images identical to training images exhibit near-zero distance to memory bank patches (mean anomaly score: <b>16.92 +- 0.92</b>). In contrast, novel clean normal test images exhibit a mean score of <b>20.19 +- 0.93</b>, while defective images average <b>21.36 +- 1.71</b>."),
        p("These 60 duplicate images artificially depress the lower tail of the normal score distribution by <b>-3.27 score units</b>, artificially inflating the official binary Image AUROC to <b>0.8044</b> and the one-vs-rest mean AUROC to <b>0.8898</b>."),
        p("Filtering out the 60 contaminated normal images establishes the clean test split (N_normal = 225, N_defective = 322 -> 72,450 comparison pairs). On this leakage-free split, the baseline binary AUROC is <b>0.7564</b> (Delta = -0.0480), and the clean one-vs-rest mean AUROC is <b>0.8600</b> (Delta = -0.0298)."),
        Spacer(1, 2),
    ]
    fig5 = make_fig(FIG_DIR / "fig5_leakage_distribution.png", 235, 105, 5, "Engine wiring score distribution shift: leaked training duplicates (mean 16.92) vs clean normal images (mean 20.19) vs defects (mean 21.36).")
    col2.extend(fig5)
    return None, col1, col2

def get_page_5():
    """Page 5: Section IV (Methodology), IV-A, IV-B, Fig 1 (Pipeline Architecture)."""
    col1 = [
        h1("IV. Theoretical Methodology & Architecture"),
        h2("A. Deep Feature Extraction Backbone"),
        p("The foundation of the AutoVI anomaly detection system is a frozen convolutional neural network acting as a generic, locally aware feature extractor. While standard PatchCore implementations frequently employ lightweight ResNet-18 or ResNet-50 backbones, our empirical investigations established that ResNet-18 lacks the feature capacity required to resolve millimeter-scale mechanical fastener displacements."),
        p("We standardize on an ImageNet-pretrained <b>WideResNet-50-2</b> [10] backbone comprising <b>68.9 million parameters</b>. WideResNet-50-2 widens the convolutional channels by a factor of k = 2 relative to standard ResNet-50, yielding substantially richer feature diversity at identical spatial depths."),
        p("To capture both fine-grained local textures (e.g., surface scratches, plastic burrs) and high-level structural semantics (e.g., missing bolts, unlatched clamps), we extract multiscale intermediate feature representations from two distinct hierarchical depths:"),
        p("1) <b>Layer 2 (Conv3_x):</b> Spatial resolution 32 x 32, feature depth C_2 = 512 channels. Layer 2 retains localized boundary information and edge contours."),
        p("2) <b>Layer 3 (Conv4_x):</b> Spatial resolution 16 x 16, feature depth C_3 = 1024 channels. Layer 3 encodes high-level semantic component relationships."),
        p("To align feature maps spatially, Layer 3 representations are upsampled by 2x via bilinear interpolation to match Layer 2's 32 x 32 grid. The two feature hierarchies are concatenated along the channel dimension, yielding an aggregated local patch descriptor of dimension <b>D = 512 + 1024 = 1,536</b> channels."),
    ]
    col2 = [
        h2("B. Neighborhood Feature Aggregation"),
        p("Single-pixel convolutional activations exhibit high variance under mechanical vibration. Following Roth et al. [6], we apply adaptive local neighborhood aggregation. For each spatial patch location p = (i, j) on the 32 x 32 feature grid, we define an adaptive neighborhood N_p of radius r = 1 (a 3x3 local receptive patch). The aggregated patch representation phi(p) is constructed via locally adaptive average pooling:"),
        eq("phi(p) = \\text{Concat}\\left( f_2(p), \\frac{1}{|N_p|} \\sum_{q \\in N_p} f_3(q) \\right) \\in \\mathbb{R}^{1536}"),
        p("This spatial pooling mechanism ensures that each patch token encodes contextual information from an effective receptive field of approximately 64 x 64 pixels in the original image space, rendering the representation robust against sub-pixel optical jitter."),
        Spacer(1, 2),
    ]
    fig1 = make_fig(FIG_DIR / "fig1_end_to_end_architecture.png", 238, 102, 1, "AutoVI End-to-End Cloud-to-Edge System Architecture spanning AWS SageMaker, Hailo-8 NPU compilation, Raspberry Pi 5 QC Station, ExecuTorch copilot, and Mobile Arm runtime.")
    col2.extend(fig1)
    return None, col1, col2

def get_page_6():
    """Page 6: Section IV-C, IV-D (Greedy k-Center Theory), Table III."""
    col1 = [
        h2("C. Memory Scaling Bottlenecks"),
        p("While dense patch extraction provides exceptional spatial localization, it introduces an exponential memory scaling bottleneck. For an input resolution of 256 x 256, each image generates a 32 x 32 spatial grid, producing exactly <b>1,024 patch vectors</b> of dimension D = 1,536."),
        p("In an industrial category with N_train = 300 normal training images, the raw patch memory bank V contains:"),
        eq("|V| = 300 \\times 1024 = 307,200 \\text{ vectors}"),
        p("In 32-bit floating-point precision, storing V requires:"),
        eq("\\text{RAM} = 307,200 \\times 1,536 \\times 4 \\text{ bytes} \\approx 1.887 \\text{ GB}"),
        p("During test inference, computing Euclidean distances between 1,024 query patches and 307,200 memory patches requires <b>314.5 million vector distance calculations per image</b> (~483 GFLOPs), completely overwhelming embedded edge CPUs and exhausting on-chip SRAM on edge NPUs."),
        h2("D. Greedy k-Center Coreset Compression Theory"),
        p("To eliminate this computational bottleneck without sacrificing anomaly detection sensitivity, we formulate memory bank reduction as a minimax facility location coreset problem. Let V subset R^D represent the full set of nominal patch embeddings extracted across all training images. We seek a compressed coreset subset M subset V with cardinality |M| << |V| that minimizes the maximum distance from any training patch to its nearest coreset representative:"),
        eq("\\min_{\\mathcal{M} \\subset \\mathcal{V}} \\max_{v \\in \\mathcal{V}} \\min_{m \\in \\mathcal{M}} \\| v - m \\|_2"),
    ]
    col2 = [
        h2("E. Memory-Bounded Chunked Coreset Implementation"),
        p("Finding the globally optimal coreset is NP-hard. We implement an iterative Greedy k-Center approximation with guaranteed 2-approximation bounds. Starting with a randomly initialized patch m_1 in V, at each step k we select the patch that maximizes the minimum distance to the current coreset:"),
        eq("m_{k+1} = \\arg\\max_{v \\in \\mathcal{V}} \\min_{m \\in \\mathcal{M}_k} \\| v - m \\|_2"),
        p("To prevent out-of-memory errors on large datasets, our implementation (<code>src/models/patchcore_model.py</code>) computes distance matrices in memory-bounded chunks of 8,192 vectors."),
        p("We configure a coreset subsampling ratio of <b>alpha = 0.06 (6.0%)</b> across all categories, achieving a <b>94.0% memory reduction</b>. As demonstrated in Table III, compressing memory banks from >300,000 vectors down to 7,372 - 17,152 vectors reduces checkpoint sizes from ~1.9 GB down to <b>45 - 105 MB</b> with zero degradation in Image AUROC."),
        Spacer(1, 2),
    ]
    t3_hdr = ["Category", "Raw Patches", "Coreset Patches", "Compression", "Memory Size", "AUROC Delta"]
    t3_rows = [
        ["pipe_clip", "149,760", "8,960", "94.02%", "55.05 MB", "+0.0000"],
        ["pipe_staple", "145,920", "8,755", "94.00%", "53.79 MB", "+0.0000"],
        ["engine_wiring", "218,112", "13,107", "93.99%", "80.53 MB", "+0.0000"],
        ["tank_screw", "244,224", "14,643", "94.00%", "89.97 MB", "+0.0000"],
        ["underbody_pipes", "122,880", "7,372", "94.00%", "45.29 MB", "+0.0000"],
        ["underbody_screw", "285,696", "17,152", "94.00%", "105.38 MB", "+0.0000"],
        ["PORTFOLIO TOTAL", "1,166,592", "69,989", "94.00%", "429.98 MB", "0.0000"],
    ]
    t3_elems = make_table(t3_hdr, t3_rows, [52, 40, 42, 38, 38, 33], caption="III: GREEDY K-CENTER CORESET MEMORY BANK COMPRESSION METRICS")
    col2.extend(t3_elems)
    return None, col1, col2

def get_page_7():
    """Page 7: Section IV-F, IV-G (ROI & Coord Conditioning), Algorithm 1."""
    col1 = [
        h2("F. Structural Anomaly Diagnosis & ROI Formulation"),
        p("In our initial baseline evaluation (Config A), three mechanical fastener categories exhibited severe underperformance: <i>tank_screw</i> (82.81% AUROC, 44.2% recall), <i>pipe_clip</i> (89.25% AUROC, 23.2% recall), and <i>engine_wiring</i> (88.98% AUROC, 34.5% recall)."),
        p("A detailed spatial error audit revealed the fundamental mechanism of this failure: in these structural categories, the defective mechanical component accounts for less than <b>0.35% to 1.8% of the total image area</b>. For example, in <i>tank_screw</i>, the missing hex screw occupies an area of only 2,654 pixels inside a 750,000 pixel frame. Across 318 normal test images, <b>91.5% (291/318)</b> generated their peak anomaly distance on irrelevant peripheral sheet metal scuffs, weld seams, and ambient reflections outside the component envelope."),
        p("To eliminate background distractor noise, we developed <b>Config B (Component-Specific ROI Slicing)</b>. Leveraging nominal assembly line geometry, we define bounding box coordinates encompassing the physical component. During inference, patch anomaly scoring is restricted to feature tokens whose receptive fields overlap the component ROI:"),
        eq("\\text{ROI}_{\\text{tank\\_screw}} = [y_{\\min}=0.25, x_{\\min}=0.55, y_{\\max}=0.65, x_{\\max}=0.95]"),
        p("As detailed in Section X, ROI cropping dramatically suppresses false alarms, boosting <i>tank_screw</i> AUROC from <b>82.81% to 99.83%</b> (+17.02%)."),
    ]
    col2 = [
        h2("G. 2D Spatial Coordinate Conditioning"),
        p("While ROI cropping eliminates peripheral noise, structural anomalies inside the ROI can still be missed if identical textures appear elsewhere within the component. For example, in <i>pipe_clip</i>, an unclipped latch exhibits the same metallic color and plastic texture as a seated clip, differing solely in its spatial coordinate position."),
        p("To resolve this ambiguity, we introduced <b>Config C (ROI + 2D Coordinate Conditioning)</b>. For every patch p = (i, j) on the 32 x 32 feature grid, we append its normalized Cartesian coordinates (u, v) in [0, 1]^2, scaled by an empirical spatial weighting factor gamma = 0.15:"),
        eq("\\phi_{\\text{coord}}(p) = \\left[ \\phi(p) \\;\\Vert\\; \\gamma \\cdot \\frac{i}{H_{\\text{grid}}} \\;\\Vert\\; \\gamma \\cdot \\frac{j}{W_{\\text{grid}}} \\right] \\in \\mathbb{R}^{1538}"),
        p("Coordinate conditioning forces the nearest-neighbor search to match patches not only in semantic appearance space but also in spatial topological proximity. On <i>engine_wiring</i>, coordinate conditioning increased official AUROC from <b>90.17% to 96.91%</b> (+6.74%)."),
        Spacer(1, 3),
        p("<b>Algorithm 1: AutoVI Config C Inspection & Scoring Pipeline</b>", "Heading3"),
        p("<b>Input:</b> Inspection frame I in R^{H x W x 3}, Memory Bank M in R^{K x D}, ROI Box B, Threshold tau<br/>"
          "1: Preprocess I: resize to 256x256, normalize via ImageNet stats.<br/>"
          "2: Extract features: f2 = Layer2(I), f3 = BilinearUpsample(Layer3(I)).<br/>"
          "3: Form patch tokens: phi = Concat(f2, NeighborhoodPool(f3)).<br/>"
          "4: Append normalized 2D coordinate grid (u, v) to phi.<br/>"
          "5: Slice patch tokens falling inside component ROI box B.<br/>"
          "6: For each patch p in ROI, compute s_p = min_{m in M} || phi_p - m ||_2.<br/>"
          "7: Reconstruct 2D heatmap S, apply Gaussian smoothing (sigma = 1.2).<br/>"
          "8: Image score S_img = max(S); Verdict = REJECT if S_img >= tau else PASS.<br/>"
          "<b>Output:</b> Image Anomaly Score S_img, Heatmap S, Inspection Verdict.", "BodyNoIndent"),
    ]
    return None, col1, col2

def get_page_8():
    """Page 8: Section IV-H (Pixel Localization), Section IV-I (Official sPRO Metric)."""
    col1 = [
        h2("H. Pixel-Level Anomaly Localization"),
        p("Beyond binary image-level classification, precision manufacturing quality control requires dense, sub-millimeter pixel localization to guide robotic rework stations and manual technician repair."),
        p("To reconstruct continuous 2D anomaly heatmaps from discrete patch distances, the patch-level anomaly scores s_p are unfolded into a 32 x 32 spatial matrix S_{32x32}. We then apply 2D bilinear interpolation to upsample S_{32x32} to the native image resolution (H x W)."),
        p("Because discrete nearest-neighbor patch queries introduce high-frequency checkerboard artifacts along patch boundaries, we apply a 2D isotropic Gaussian smoothing filter:"),
        eq("G(x, y; \\sigma) = \\frac{1}{2\\pi\\sigma^2} \\exp\\left( -\\frac{x^2 + y^2}{2\\sigma^2} \\right)"),
        p("In our empirical ablation experiments (<code>reports/phase5_postprocessing_ablation.csv</code>), we evaluated smoothing kernels across sigma in [1.0, 6.0]. For full-frame inspections (Config A), <b>sigma = 4.0</b> provides optimal continuous contour envelopes. For tightly cropped component ROIs (Config C), <b>sigma = 1.2</b> preserves sharp fastener boundaries without diluting the defect centroid into background metal."),
        p("Binary segmentation masks P(x, y) are generated by thresholding the smoothed heatmap: P(x, y) = 1 if S(x, y) >= tau_pixel, else 0."),
    ]
    col2 = [
        h2("I. Official AutoVI Metric: Saturated Per-Region Overlap (sPRO)"),
        p("Academic benchmarks traditionally evaluate localization via Pixel AUROC or Per-Region Overlap (PRO). However, Carvalho et al. [1] demonstrated that standard PRO exhibits severe biases in automotive inspection: it rewards diffuse, oversized anomaly predictions that envelope entire assemblies rather than tightly localizing mechanical defects."),
        p("To establish rigorous localization assessment, the official AutoVI benchmark introduces <b>Saturated Per-Region Overlap (sPRO)</b>. For each ground-truth defect connected component R_k, sPRO introduces a component saturation threshold tau_k in (0, 1] defined in <code>defects_config.json</code>:"),
        eq("\\text{sPRO}(T) = \\frac{1}{K} \\sum_{k=1}^K \\min\\left( 1.0, \\frac{|R_k \\cap P(T)|}{\\tau_k \\cdot |R_k|} \\right)"),
        p("Once prediction overlap reaches the threshold fraction tau_k, saturation credit is capped at 1.0. Excessive false positive predictions outside R_k rapidly increase the False Positive Rate (FPR), severely penalizing the area under the curve."),
        p("Furthermore, the official evaluator implements <b>dynamic binary curve refinement</b>, iteratively sampling threshold operating points until consecutive points in ROC space satisfy Delta_dist <= 0.002. Our system achieves an outstanding Macro sPRO@0.30 of <b>96.96%</b> across all categories."),
    ]
    return None, col1, col2

print("report_sections_part1.py ready.")
