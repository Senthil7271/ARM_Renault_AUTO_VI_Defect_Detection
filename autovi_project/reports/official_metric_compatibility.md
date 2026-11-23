# AutoVI Official Benchmark Metric Compatibility Analysis

This report documents the architectural, mathematical, and algorithmic differences between the **existing PatchCore evaluation pipeline** (`autovi_project`) and the **official AutoVI evaluation repository** (`external/autovi_evaluation_code`, commit `8ccc3d95054edf8237a152a6c29f61c0504b0b09`).

---

## 1. Metric & Methodology Compatibility Matrix

| Evaluation Dimension | Existing PatchCore Implementation | Official AutoVI Benchmark (`phcarval/autovi_evaluation_code`) | Compatible? | Required Adaptation / Correction |
| :--- | :--- | :--- | :---: | :--- |
| **Primary Localization Metric** | **AUPRO (0.3)** (Unsaturated connected-component per-region overlap up to FPR 0.3) | **AUC-sPRO (0.01, 0.05, 0.1, 0.3, 1.0)** (Saturated per-region overlap with defect saturation thresholds) | **Partial (Different variant)** | Retain existing AUPRO as supplemental; evaluate official AUC-sPRO across all integration limits using official codebase. |
| **Primary Classification Metric** | **Image AUROC** (Standard ROC curve across all test samples: normal vs defective) | **Subtype-Averaged AUC-ROC** (Computed per defect subtype vs good, then macro-averaged) | **Different Aggregation** | Existing computes global ROC; official defines macro-averaged ROC per subtype (defined in script, though omitted from default `main` JSON output). |
| **Pixel AUROC** | Standard pixel-wise ROC AUC across all test pixels | **Not implemented in official benchmark** | **Supplemental only** | Official benchmark omits Pixel AUROC because background dominance skews the metric; retain Pixel AUROC as supplemental industrial metric. |
| **Ground Truth Channels** | Flat 2D binary union (`np.bitwise_or` of all defect masks) | Multi-channel `GroundTruthMap` (individual channels per defect instance) | **Architecturally Distinct** | For sPRO, official evaluates each mask channel independently with its specific `saturation_threshold`. |
| **Defect Saturation** | No saturation: linear area ratio $TP / Area$ | Saturated: $\min(TP / SaturationArea, 1.0)$, where $SaturationArea = \tau \times Area$ | **Novel to AutoVI** | Defined in `defects_config.json` (e.g., $\tau=0.5$ for `unclipped`, $\tau=1.0$ for `operator`). |
| **Evaluation Grid Size** | Normalized $256 \times 256$ square grid | Object-specific resize shape: $(400, 400)$ for wiring/clips; $(1000, 750)$ for screws/pipes | **Resolution Difference** | Export full-resolution anomaly maps so the official evaluator can resize them to the canonical benchmark shape. |
| **Threshold Sampling** | Fixed discrete threshold sweep (e.g. 500 linearly spaced steps) | Adaptive binary refinement (`binary_refinement`, $\Delta \le 0.001$) | **Different Numerical Precision** | Official evaluator dynamically samples thresholds to enforce curve smoothness. |
| **OS Compatibility** | Cross-platform (Windows & Linux, pure Python/PyTorch) | Linux-only assumption (`os.nice(19)` in `evaluate_experiment.py`) | **Incompatible on Windows** | Wrap execution in non-intrusive adapter that shims `os.nice` when running on Windows. |

---

## 2. In-Depth Mathematical Diagnoses

### 2.1 Saturated Per-Region Overlap (sPRO) vs. Standard PRO
- **Standard PRO**: For each ground-truth connected component $C_k$, the overlap at threshold $\theta$ is:
  $$\text{PRO}_k(\theta) = \frac{|P_\theta \cap C_k|}{|C_k|}$$
  and the mean PRO is integrated with respect to false positive rate (FPR) up to $\text{FPR}_{\max} = 0.3$.
- **Official AutoVI sPRO**: Adapted from MVTec LOCO AD. For ground truth defect channel $C_k$ with defect configuration $\tau_k \in (0, 1]$:
  $$\text{sPRO}_k(\theta) = \min\left(1.0, \frac{|P_\theta \cap C_k|}{\tau_k |C_k|}\right)$$
  When an anomaly map covers at least $\tau_k$ of the defect region (e.g., 50% for `unclipped`), it receives full credit (1.0). This accounts for industrial assembly defects where partial localization of a detached clip or misrouted wire is sufficient for quality alerting.

### 2.2 Image Anomaly Score Derivation
- **PatchCore Formulation**: Computes patch-level distance to nearest memory-bank neighbors and weights the maximum distance by neighbor dispersion:
  $$s^*(x) = \left(1 - \frac{\exp(\|\tilde{m}^* - m^{(2)}\|)}{\sum_{i=1}^k \exp(\|\tilde{m}^* - m^{(i)}\|)}\right) \cdot \|\tilde{m}^* - m^{(1)}\|$$
- **Official Evaluator Protocol**: Expects image-level score to be:
  $$s_{\text{image}} = \max_{(i, j)} A(i, j)$$
  where $A$ is the 2D pixel anomaly map. If anomaly maps are independently min-max normalized per image for visualization, this completely destroys inter-image calibration!
- **Engineering Rule**: Anomaly maps exported for official evaluation must preserve **global unnormalized scale** across the test set.

### 2.3 Resizing and Aspect Ratios
- Categories `pipe_clip`, `engine_wiring`, and `pipe_staple` originate as square camera crops and are evaluated at **$400 \times 400$**.
- Categories `tank_screw`, `underbody_pipes`, and `underbody_screw` originate from high-resolution underbody line cameras and are evaluated at **$1000 \times 750$** (aspect ratio 4:3).
- Our PatchCore backbone operates on $256 \times 256$ tensors. When exporting anomaly maps, bi-cubic or bi-linear interpolation restores the maps to the canonical evaluation resolutions.

---

## 3. Discovered Source Code Defects in Official Evaluator

During audit of `external/autovi_evaluation_code/` (commit `8ccc3d95054edf8237a152a6c29f61c0504b0b09`), four notable issues were identified:
1. **Windows Platform Incompatibility**: `evaluate_experiment.py` calls `set_niceness(args.niceness)`, invoking `os.nice()`. On Windows systems, this throws `AttributeError: module 'os' has no attribute 'nice'`.
2. **Missing Import in Batch Wrapper**: `evaluate_multiple_experiments.py` line 17 imports `from evaluate_experiment import OBJECT_NAMES`, but `OBJECT_NAMES` was deleted from `evaluate_experiment.py`. Running `evaluate_multiple_experiments.py` fails with an `ImportError`.
3. **Missing Tool**: `print_metrics.py` is referenced in the documentation but was omitted from the repository commit.
4. **Omission of Classification Metric**: `get_image_level_detection_metrics` is implemented in `evaluate_experiment.py`, but `main()` never invokes it, outputting only `localization` in `metrics.json`.
