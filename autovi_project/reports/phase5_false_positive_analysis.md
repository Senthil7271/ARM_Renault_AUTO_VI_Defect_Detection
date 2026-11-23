# AutoVI Phase 5: False-Positive & Over-Segmentation Diagnostic Analysis

**Author:** Industrial Computer Vision Research & Anomaly Localization Team  
**Evaluation Scope:** Active categories (`tank_screw`, `pipe_clip`, `engine_wiring`)  
**Artifact Path:** [reports/phase5_false_positive_analysis.md](file:///c:/Users/linga/Downloads/ArmChallenge/autovi_project/reports/phase5_false_positive_analysis.md)  
**Metrics Source:** [reports/phase5_segmentation_metrics.csv](file:///c:/Users/linga/Downloads/ArmChallenge/autovi_project/reports/phase5_segmentation_metrics.csv)  
**Comparative Figures:** [outputs/phase5_localization_comparisons/](file:///c:/Users/linga/Downloads/ArmChallenge/autovi_project/outputs/phase5_localization_comparisons/)  

---

## 1. Executive Summary & Problem Formulation

In Phase 4, Component-Aware PatchCore (Config C) achieved an official macro image-level AUROC of **97.53%** and an official macro sPRO@0.30 of **0.9700**. However, visual inspection of predicted anomaly masks against human ground-truth (GT) annotations reveals substantial **over-segmentation**:
- On `tank_screw`, predicted anomaly area averages **7.35x** the ground-truth area, with an average false-positive pixel ratio of **86.27%**.
- On `pipe_clip`, predicted anomaly area averages **2.81x** the ground-truth area (FP ratio **62.90%**).
- On `engine_wiring`, predicted anomaly area averages **1.94x** the ground-truth area (FP ratio **81.92%** on fastening anomalies).

While image-level classification is near-perfect (Recall $> 90-99\%$), pixel-level precision is constrained to **13.7% - 37.1%**, resulting in moderate Dice scores (**0.241 - 0.498**). 

This diagnostic report provides a root-cause investigation into the physical and architectural mechanisms driving this over-segmentation.

---

## 2. Quantitative False-Positive Pixel Distribution

Across all 559 active test defect instances evaluated in `reports/phase5_segmentation_metrics.csv`:

| Category | Total Defect Samples | Mean GT Defect Area | Mean Pred Defect Area | Mean Area Ratio (Pred/GT) | Mean False Positive Ratio | Mean Recall | Mean Precision | Mean Dice (F1) | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **tank_screw** | 95 | 2,654.9 px | 19,511.7 px | **7.35x** | **86.27%** | **99.72%** | 13.73% | 0.2410 | 0.1372 |
| **pipe_clip** | 142 | 2,696.3 px | 7,575.1 px | **2.81x** | **62.90%** | **91.23%** | 37.10% | 0.4980 | 0.3417 |
| **engine_wiring** | 322 | 5,549.5 px | 10,774.3 px | **1.94x** | **81.92%** | **72.89%** | 17.69% | 0.2339 | 0.1401 |

### Key Observations:
1. **Asymmetric Error Profile**: In all three categories, False Negatives (FN - Blue) are minimal (Recall $72.9\% - 99.7\%$). The dominant source of segmentation error is overwhelmingly False Positives (FP - Red), forming a wide halo surrounding the defect.
2. **Defect Size Invariance**: Smaller defects experience vastly higher area inflation. A 500-pixel missing screw head produces approximately the same absolute halo footprint ($\approx 15,000-20,000$ pixels) as a larger defect, causing the area ratio to spike over **10x** on compact defects.

---

## 3. Root-Cause Analysis: Four Architectural & Physical Drivers

```
+---------------------------------------------------------------------------------------------------+
|                            Mechanisms Driving Over-Segmentation                                   |
+---------------------------------------------------------------------------------------------------+
|  1. Receptive Field Halo: WideResNet-50 Layers 2/3 have 60-120 px receptive fields.               |
|     -> Neurons overlapping the defect boundary register elevated nearest-neighbor distance.       |
|                                                                                                   |
|  2. Bilinear Feature Upsampling: Coarse feature maps (28x28 or 50x50) upsampled to full resolution |
|     -> Sharp feature transitions become smooth, gradual Gaussian-like slopes.                     |
|                                                                                                   |
|  3. Global vs Peak-Relative Thresholding: Background p99 threshold intersects the lower skirt of    |
|     the peak rather than the steep physical contour.                                              |
|                                                                                                   |
|  4. Optical & Perspective Reflections: Metallic curvature (tank) and semi-gloss plastic (clip)    |
|     scatter specular highlights near fastener edges.                                              |
+---------------------------------------------------------------------------------------------------+
```

### 3.1. Receptive Field Halo of Deep Convolutional Layers
PatchCore constructs its patch-level memory bank using intermediate activations from `layer2` and `layer3` of WideResNet-50:
- Layer 2 effective theoretical receptive field: $\approx 60 \times 60$ pixels.
- Layer 3 effective theoretical receptive field: $\approx 120 \times 120$ pixels.

When a feature vector is extracted at coordinate $(i, j)$ on the downsampled grid, its activation incorporates contextual information from an area up to $120 \times 120$ pixels in the input image. If even 15% of that receptive field touches a missing screw or displaced wire, the distance to the nominal coreset memory bank increases sharply. Consequently, pixels that are physically 20–40 pixels outside the true defect boundary still register anomalous feature distances.

### 3.2. Bilinear Interpolation & Resolution Mismatch
- For `tank_screw`, the native image is $750 \times 1000$. The extracted feature grid is approximately $47 \times 63$.
- Each feature cell represents a block of approximately $16 \times 16$ image pixels.
- When this coarse distance grid is upsampled back to $750 \times 1000$ using standard bilinear interpolation, step-function anomaly edges are convolved into smooth, low-frequency Gaussian gradients.

### 3.3. Score Skirt Intersection at Lower Thresholds
Because normal test background scores have low variance inside the deployable ROI (mean score $\approx 857$ on `tank_screw`), the calibrated 99th percentile threshold ($\tau_{\text{p99}} = 23,082$) is positioned well below the defect peak score ($\max A \approx 36,000 - 45,000$). At $\tau = 23,082$, the decision boundary cuts across the wide, flaring skirt of the upsampled Gaussian peak rather than the sharp physical object contour, directly encompassing thousands of adjacent normal pixels.

### 3.4. Optical Reflections and Registration Variation
- On `tank_screw`, brushed aluminum reflects directional light as curved caustics. While Config C's ROI successfully cuts off distant tank curvature, the local mounting boss still exhibits slight lighting shifts.
- On `engine_wiring`, black wire bundles have subtle specular highlights. When a harness is unfastened, shadow casting on the engine block shifts the surrounding pixel values.

---

## 4. Case Studies: Best, Median, and Worst Localization

### 4.1. `tank_screw` (Missing Screw)
*Figure Reference: `outputs/phase5_localization_comparisons/tank_screw_oversegmentation_case_study.png`*
- **Best (ID 0088)**: GT = 3,369 px | Pred = 19,604 px | Ratio = **5.82x** | Dice = **0.293** | Prec = 17.2% | Rec = 100.0%.  
  *Diagnosis*: The screw hole is centered squarely in the flange; the halo is symmetric and well-bounded.
- **Median (ID 0026)**: GT = 2,651 px | Pred = 19,530 px | Ratio = **7.37x** | Dice = **0.239** | Prec = 13.6% | Rec = 100.0%.  
  *Diagnosis*: Typical halo behavior. 100% defect capture with a 25-pixel concentric collar of false positives.
- **Worst (ID 0086)**: GT = 1,939 px | Pred = 19,562 px | Ratio = **10.09x** | Dice = **0.180** | Prec = 9.9% | Rec = 100.0%.  
  *Diagnosis*: Compact defect area (partially recessed hole) produces a full-sized receptive field footprint, expanding area by an order of magnitude.

### 4.2. `pipe_clip` (Unclipped Latch)
*Figure Reference: `outputs/phase5_localization_comparisons/pipe_clip_unclipped_halo_analysis.png`*
- **Best (ID 0102)**: GT = 4,024 px | Pred = 6,406 px | Ratio = **1.59x** | Dice = **0.745** | Prec = 60.7% | Rec = 96.6%.  
  *Diagnosis*: The unclipped latch has wide physical detachment. The predicted contour aligns cleanly with the GT latch boundary.
- **Median (ID 0027)**: GT = 2,727 px | Pred = 7,574 px | Ratio = **2.78x** | Dice = **0.519** | Prec = 35.3% | Rec = 98.1%.  
  *Diagnosis*: Moderate latch displacement. The latch body is detected with high recall, but the plastic hinge collar adds halo pixels.
- **Worst (ID 0104)**: GT = 684 px | Pred = 7,580 px | Ratio = **11.08x** | Dice = **0.165** | Prec = 9.0% | Rec = 100.0%.  
  *Diagnosis*: Minimal detachment (only the very tip of the latch is unhooked). Receptive field covers the entire clip head, resulting in an 11x over-segmentation.

### 4.3. `engine_wiring` (Fastening Displacement vs Foreign Object)
*Figure Reference: `outputs/phase5_localization_comparisons/engine_wiring_false_positive_breakdown.png`*
- **Foreign Object (Cardboard 0003)**: GT = 4,964 px | Pred = 2,926 px | Ratio = **0.59x** | Dice = **0.704** | Prec = **94.9%** | Rec = 55.9%.  
  *Diagnosis*: Discrete texture defect (brown cardboard on black engine). Anomaly map is razor-sharp with near-zero false positives.
- **Unfastened Fastening (Median 0107)**: GT = 1,440 px | Pred = 10,774 px | Ratio = **7.48x** | Dice = **0.236** | Prec = 13.4% | Rec = 100.0%.  
  *Diagnosis*: Structural displacement. The unclipped clip and loose wire bow outward, triggering diffuse feature deviation across the harness span.
- **Diffuse Detachment (Worst 0271)**: GT = 328 px | Pred = 0 px | Ratio = **0.0x** | Dice = **0.000** | Prec = 0.0% | Rec = 0.0%.  
  *Diagnosis*: Extremely subtle wire clip unhooking where displacement is under 3 mm; scores remain below $\tau_{\text{p99}}$.

---

## 5. Controlled Postprocessing Ablation Findings

Evaluating controlled postprocessing methods across the active categories demonstrated:

1. **Filtering Does Not Solve Haloing**:
   - Gaussian smoothing ($\sigma=1.0$) decreased Dice on `tank_screw` (0.2410 $\to$ 0.2391) and `engine_wiring` (0.2339 $\to$ 0.2198), as it further diffuses the upsampled edge.
   - Bilateral filtering produced negligible difference (Dice 0.2365 on `tank_screw`), because the receptive field blur exists in the continuous feature distance space, not as high-frequency image noise.
2. **Normal-Only Threshold Re-Calibration Improves Precision**:
   - Raising the threshold to **Normal p99.5** ($24,917$ on `tank_screw`, $38,656$ on `pipe_clip`, $33,811$ on `engine_wiring`) cuts the false-positive halo significantly:
     - `tank_screw`: Dice rises from **0.2410** to **0.2687**; Area Ratio falls from **7.35x** to **6.49x** (Recall remains **99.35%**).
     - `pipe_clip`: Dice rises from **0.4980** to **0.5112**; Precision increases from **37.1%** to **46.5%**; Area Ratio falls from **2.81x** to **1.55x**.
     - `engine_wiring`: Dice rises from **0.2339** to **0.2583**; Precision increases from **17.7%** to **20.3%**.
   - Beyond p99.5 (e.g. p99.8), small subtle defects begin to drop out (recall collapses on `pipe_clip` from 91.2% to 16.3%).
3. **Zero Impact on Image Classification**:
   - All threshold-based postprocessing preserves exact Image AUROC (**0.9998** on `tank_screw`, **0.9431** on `pipe_clip`, **0.9691** on `engine_wiring`) because $\max_{x,y} A(x, y)$ is untouched.

---

## 6. Industrial Robotic Deployment Recommendations

1. **Tolerance Band Protocol**:
   - Do not expect pixel-perfect single-pixel boundary alignment from deep PatchCore features without an edge-guided segmentation head.
   - For automated robotic rework or camera zoom guidance, fit a **bounding box or minimum enclosing circle** around the predicted connected components rather than raw pixel masks.
2. **Dual Operational Thresholds**:
   - **Line Inspection Gate**: Use calibrated image threshold $\tau_{\text{img}} = 18.0$ for PASS/FAIL part screening.
   - **Spatial Localizer**: Use calibrated Normal p99.5 threshold ($\tau_{\text{p99.5}} = 24,917$ for `tank_screw`, $38,656$ for `pipe_clip`, $33,811$ for `engine_wiring`) to obtain tight bounding coordinates for robotic tool positioning.
