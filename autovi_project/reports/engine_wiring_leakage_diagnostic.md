# Engine Wiring Train/Test Leakage Diagnostic Report

Quantitative analysis of train-to-test data leakage caused by 60 identical normal images shared between `train/good` and `test/good` in the official AutoVI dataset.

## 1. Exact Sample Counts and Evaluation Denominators

- **Total Official Test Samples**: 607 images
  - Official Test Normal (`good`): **285** images
  - Official Test Defective (`anomaly`): **322** images
  - Official ROC Comparison Pairs ($N_{\text{norm}} \times N_{\text{anom}}$): **91,770** pairs
- **Unique Contaminated Test-Normal Samples**: **60** images (identified via identical SHA-256 hashes matching 60 images in `train/good`)
- **Contaminated Defective Samples**: **0** images (leakage is strictly confined to normal images)
- **Clean (Leakage-Free) Test Split**: **547** images
  - Clean Test Normal (`good`): **225** images ($285 - 60 = 225$)
  - Clean Test Defective (`anomaly`): **322** images (100% retained)
  - Clean ROC Comparison Pairs ($N_{\text{clean\_norm}} \times N_{\text{anom}}$): **72,450** pairs

## 2. Quantitative Metric Comparison

| Evaluation Split | Normal ($N$) | Defective ($N$) | ROC Pairs | Binary Image AUROC | One-vs-Rest Mean AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Official Benchmark Split** | 285 | 322 | 91,770 | **0.8044** | **0.8898** |
| **Leakage-Free Clean Split** | 225 | 322 | 72,450 | **0.7564** | **0.8600** |
| **Leakage Impact ($\Delta$)** | -60 | 0 | -19,320 | **-0.0480** | **-0.0298** |

### Per-Defect Subtype Clean AUROC Breakdown
- `blue_hoop` ($N = 5$ vs. $225$ clean normal): Clean AUROC = **0.8133** (Official: 21.33 mean score)
- `cardboard` ($N = 5$ vs. $225$ clean normal): Clean AUROC = **1.0000** (Official: 30.03 mean score)
- `fastening` ($N = 277$ vs. $225$ clean normal): Clean AUROC = **0.7278** (Official: 21.05 mean score)
- `multiple` ($N = 33$ vs. $225$ clean normal): Clean AUROC = **0.9479** (Official: 22.75 mean score)
- `obstruction` ($N = 2$ vs. $225$ clean normal): Clean AUROC = **0.8111** (Official: 21.07 mean score)

## 3. Score Distribution Analysis

- **Leaked Normal Images** ($N = 60$): Mean = **16.9167** ($\pm 0.9193$)
- **Clean Normal Images** ($N = 225$): Mean = **20.1858** ($\pm 0.9335$)
- **Defective Images** ($N = 322$): Mean = **21.3645** ($\pm 1.7109$)

### Scientific Interpretation
Because PatchCore evaluates test samples by computing minimum Euclidean distances to the nearest training patch embeddings, test images identical to training images exhibit artificially lower nominal distances (mean: 16.92 vs. clean: 20.19, a $\Delta = -3.27$ score difference). This shifts the lower tail of the normal score distribution downward, creating an artificial separation from anomalous samples and inflating official Image AUROC by **+0.0480** (binary) and **+0.0298** (one-vs-rest mean). Removing these 60 duplicate normal images establishes the true, unbiased generalization baseline of the model on novel engine components.
