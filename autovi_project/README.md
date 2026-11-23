# AutoVI Industrial Open-Set Anomaly Detection (PatchCore + WideResNet-50-2)

Production-grade industrial anomaly detection system designed for the **Automotive Visual Inspection Dataset (AutoVI)**, developed by Renault Group and Université de technologie de Compiègne.

---

## 1. System Overview

This system learns strictly from **defect-free (normal) industrial images** to detect, score, and localize abnormal appearances not present during training.

### Key Capabilities
- **Architecture**: PatchCore with pretrained `WideResNet-50-2` (layers `layer2` + `layer3`).
- **Normal-Only Training**: Strict `train/good/` learning with reproducible 90/10 validation calibration.
- **Image-Level Scoring**: Max patch distance with calibrated normal/anomaly thresholding.
- **Pixel-Level Localization**: Aligned patch feature maps with Gaussian smoothing ($\sigma=4$) and thresholded binary defect masks.
- **Multi-Mask Support**: Robust bitwise OR combination of multiple ground-truth masks per image.
- **Open-Set Event Logging**: Structured JSON events for abnormal products, keeping `defect_class: "unknown"` in accordance with unsupervised principles.
- **Independent 6-Category Support**: Isolated memory banks for `pipe_clip`, `engine_wiring`, `pipe_staple`, `tank_screw`, `underbody_pipes`, and `underbody_screw`.
- **Edge Deployment Roadmap**: Hailo-8 AI accelerator deployment feasibility report.

---

## 2. Directory Layout

```
ArmChallenge/
├── engine_wiring/
├── pipe_clip/
├── pipe_staple/
├── tank_screw/
├── underbody_pipes/
├── underbody_screw/
│
└── autovi_project/
    ├── configs/
    │   ├── default.yaml            # PatchCore and data hyperparameters
    │   └── categories.yaml         # Category metadata and expected counts
    │
    ├── src/
    │   ├── dataset/
    │   │   ├── audit.py            # Integrity, hashing & leakage auditor
    │   │   ├── loader.py           # Normal-only datamodule & split manifest
    │   │   └── masks.py            # Multi-mask loader & NN resizer
    │   ├── models/
    │   │   └── patchcore_model.py  # PatchCore, hooks, coreset & memory bank
    │   ├── training/
    │   │   └── trainer.py          # Training loop & threshold calibration
    │   ├── evaluation/
    │   │   ├── metrics.py          # Image/Pixel AUROC, AP, AUPRO, F1, IoU
    │   │   └── evaluator.py        # Test split evaluator & curve plotter
    │   ├── inference/
    │   │   ├── predictor.py        # Single image & batch predictor
    │   │   └── events.py           # Anomaly event JSON logger
    │   └── visualization/
    │       └── heatmaps.py         # 4-panel inspection comparisons
    │
    ├── scripts/
    │   ├── audit_dataset.py        # Run dataset audit
    │   ├── train.py                # Train category model
    │   ├── evaluate.py             # Evaluate test split
    │   ├── predict.py              # Run inference on test image
    │   └── benchmark.py            # Latency and throughput benchmark
    │
    ├── tests/
    │   ├── test_dataset.py         # Dataset detection and split tests
    │   ├── test_masks.py           # Mask pairing & multi-mask union tests
    │   ├── test_metrics.py         # AUROC, AP, AUPRO, IoU metric tests
    │   └── test_patchcore.py       # Coreset, persistence & event tests
    │
    ├── outputs/
    │   ├── dataset_audit/          # 6 audit CSVs + summary markdown
    │   └── experiments/            # Trained models, metrics & visualizations
    │
    ├── reports/
    │   ├── dataset_audit.md        # Comprehensive dataset audit report
    │   ├── hailo_feasibility.md    # Edge Hailo-8 deployment report
    │   └── six_category_comparison.csv
    │
    ├── requirements.txt
    ├── environment_report.md
    └── README.md
```

---

## 3. Terminal Execution Guide

You can run the entire pipeline directly in your PowerShell terminal using Python 3.10.

### Step 0: Set Python Alias / Environment
```powershell
$py = "C:\Users\linga\AppData\Local\Programs\Python\Python310\python.exe"
```

### Step 1: Run Automated Test Suite
```powershell
& $py -m unittest discover -s autovi_project/tests -p "test_*.py" -v
```
*(All 14 unit tests pass verifying dataset loading, masks, coreset subsampling, persistence, and metrics)*

### Step 2: Run Dataset Audit
```powershell
& $py autovi_project/scripts/audit_dataset.py
```
*(Generates `dataset_summary.csv`, `defect_distribution.csv`, `image_dimensions.csv`, `mask_validation.csv`, `duplicate_report.csv`, and `data_leakage_report.csv` in `autovi_project/outputs/dataset_audit/`)*

### Step 3: Train PatchCore Model on `pipe_clip`
```powershell
& $py autovi_project/scripts/train.py --category pipe_clip
```
To optimize for CPU memory and speed, you can configure the coreset ratio and chunk size:
```powershell
# Fast CPU test with 1% coreset ratio (1,792 vectors, ~22s selection time):
& $py autovi_project/scripts/train.py --category pipe_clip --coreset-ratio 0.01

# Balanced 5% coreset ratio (8,960 vectors):
& $py autovi_project/scripts/train.py --category pipe_clip --coreset-ratio 0.05

# Full 10% coreset ratio (17,920 vectors):
& $py autovi_project/scripts/train.py --category pipe_clip --coreset-ratio 0.10 --chunk-size 8192
```

### Step 4: Evaluate Model on Official Test Set
```powershell
& $py autovi_project/scripts/evaluate.py --category pipe_clip --visualize-count 8
```
*(Produces `metrics.json`, `predictions.csv`, `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, and 4-panel comparison images in `outputs/experiments/pipe_clip/visualizations/`)*

### Step 5: Run Single-Image Inference & Event Logging
```powershell
& $py autovi_project/scripts/predict.py --category pipe_clip --image "pipe_clip/pipe_clip/test/unclipped/0000.png"
```

### Step 6: Benchmark Latency & Throughput
```powershell
& $py autovi_project/scripts/benchmark.py --category pipe_clip --runs 30
```

### Step 7: Train and Evaluate All 6 Categories
```powershell
& $py autovi_project/scripts/train.py --category all
& $py autovi_project/scripts/evaluate.py --category all
```
*(Generates comparative benchmark in `autovi_project/reports/six_category_comparison.csv`)*
