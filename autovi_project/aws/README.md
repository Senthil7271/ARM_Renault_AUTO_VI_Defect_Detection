# AutoVI PatchCore — AWS SageMaker GPU Training & Deployment

This module provides a production-grade, headless migration for training and evaluating **PatchCore + WideResNet-50-2** on **AWS SageMaker GPU instances** (`ml.g4dn.xlarge` with NVIDIA T4 16GB VRAM, CUDA 12.1).

---

## 1. System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Local Windows Workspace                         │
│  - autovi_project/src/ (PatchCore PyTorch model, loader, metrics)      │
│  - autovi_project/aws/ (Deployment, packaging, verification scripts)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ upload_code.py
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Amazon S3 Storage Bucket                         │
│  s3://sagemaker-us-east-1-077578771161/                                │
│  ├── autovi/                    <-- Full AutoVI dataset (4,858 images) │
│  │   ├── engine_wiring/                                                │
│  │   ├── pipe_clip/                                                    │
│  │   └── ...                                                           │
│  ├── autovi/deployment/                                                │
│  │   └── sourcedir.tar.gz       <-- Packaged code & entry point        │
│  └── autovi/outputs/                                                   │
│      └── autovi-patchcore-gpu-.../                                     │
│          ├── output/output.tar.gz  <-- Evaluation metrics, heatmaps    │
│          └── output/model.tar.gz   <-- Trained memory bank & weights   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ submit_training.py
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Amazon SageMaker Training (ml.g4dn.xlarge)               │
│  Container: PyTorch 2.1 GPU DLC (CUDA 12.1, Python 3.10, Ubuntu 20.04)  │
│  Execution: sagemaker_entry.py                                         │
│  1. GPU WideResNet-50-2 feature extraction (inference_mode)            │
│  2. Memory-bounded greedy k-center coreset subsampling                 │
│  3. Validation calibration (target FPR = 0.05)                         │
│  4. Full test set evaluation (Pixel AUROC, AUPRO, Image AUROC)         │
│  5. Anomaly heatmap & segmentation visualization generation            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```text
autovi_project/aws/
├── README.md                      # This deployment guide
├── requirements-training.txt       # Dependencies for SageMaker DLC
├── configs/
│   └── sagemaker.yaml             # Cluster, S3, container & hyperparam configs
├── tests/
│   └── test_aws_entry.py          # Unit tests for entrypoint & packaging
├── sagemaker_entry.py             # SageMaker container entrypoint
├── verify_aws.py                  # Pre-flight credential, S3 & API verifier
├── upload_code.py                 # Packages src/ & uploads sourcedir.tar.gz
├── submit_training.py             # Launches SageMaker training jobs via boto3
├── download_results.py            # Downloads model.tar.gz & output.tar.gz
└── validate_artifacts.py          # Verifies downloaded model on local CPU
```

---

## 3. Cost & Resource Optimization ($50 Credit Budget)

| Parameter | Local CPU Baseline | SageMaker GPU (`ml.g4dn.xlarge`) |
| :--- | :--- | :--- |
| **Compute Device** | Intel Core CPU | NVIDIA T4 (16 GiB VRAM, Tensor Cores) |
| **Hourly Cost** | \$0.00 (Local) | **\$0.736 / hour** in `us-east-1` |
| **Max Runtime** | 10–20 min / category | **3–5 min / category** |
| **Job Cost Estimate** | N/A | **~\$0.04 – \$0.06 per category** |
| **Full 6-Category Run**| ~2 hours CPU load | **~\$0.30 total (< 1% of \$50 credit)** |
| **Auto-Shutdown** | Manual | Built-in `MaxRuntimeInSeconds: 7200` |

---

## 4. IAM Permissions & Prerequisites

To execute SageMaker training from your local terminal using AWS Profile `autovi`:

### Prerequisites Checklist:
1. **AWS CLI configured**: Profile `autovi` linked to IAM User `autovi-s3-uploader`.
2. **S3 Dataset uploaded**: `s3://sagemaker-us-east-1-077578771161/autovi/` contains all categories.
3. **IAM Permissions for local user `autovi-s3-uploader`**:
   To call `sagemaker:CreateTrainingJob` and pass the execution role, the IAM user needs:
   - Policy: `AmazonSageMakerFullAccess` (or custom policy with `sagemaker:CreateTrainingJob`, `sagemaker:DescribeTrainingJob`, `sagemaker:ListTrainingJobs`).
   - Policy: `iam:PassRole` granting permission to pass `arn:aws:iam::077578771161:role/service-role/AmazonSageMaker-ExecutionRole-20261115T084659` to SageMaker.

*(Note: If running directly inside an **AWS SageMaker Studio Notebook**, the notebook's pre-attached execution role already possesses all required permissions!)*

---

## 5. Step-by-Step Execution Workflow

Set your Python executable path in PowerShell:
```powershell
$py = "C:\Users\linga\AppData\Local\Programs\Python\Python310\python.exe"
```

### Step 1: Pre-flight AWS Verification
Run the verification script to check AWS STS identity, S3 access, and SageMaker API availability:
```powershell
& $py autovi_project/aws/verify_aws.py
```

### Step 2: Package and Upload Source Code
Packages `src/`, `configs/`, `sagemaker_entry.py`, and `requirements-training.txt` into `sourcedir.tar.gz` and pushes it to S3:
```powershell
& $py autovi_project/aws/upload_code.py
```

### Step 3: Launch SageMaker GPU Training Job
Launch a smoke test on `pipe_clip` with 5% coreset ratio:
```powershell
& $py autovi_project/aws/submit_training.py --category pipe_clip --coreset-ratio 0.05
```

For a fast smoke test (1% coreset):
```powershell
& $py autovi_project/aws/submit_training.py --category pipe_clip --coreset-ratio 0.01
```

For all 6 categories (once smoke test succeeds):
```powershell
& $py autovi_project/aws/submit_training.py --category all --coreset-ratio 0.05
```

Options:
- `--wait`: Keeps the script running and streams CloudWatch logs until completion.
- `--no-wait`: Submits the job and exits immediately (check status later).

### Step 4: Download Results & Artifacts
Once the job transitions to `Completed`:
```powershell
& $py autovi_project/aws/download_results.py --job-name <TRAINING_JOB_NAME>
```
Artifacts are automatically unpacked into:
`outputs/sagemaker_runs/<TRAINING_JOB_NAME>/`
- `model/patchcore_model.pkl` (Trained memory bank & calibrated threshold)
- `output/test_evaluation_report.json` (Pixel AUROC, AUPRO, Image AUROC, latency)
- `output/visualizations/` (4-panel anomaly segmentation overlays)

### Step 5: Validate Downloaded Model Locally on CPU
Verify that the model trained on the cloud GPU loads and functions seamlessly on local CPU without errors:
```powershell
& $py autovi_project/aws/validate_artifacts.py --run-dir outputs/sagemaker_runs/<TRAINING_JOB_NAME>
```

---

## 6. Running Tests

To verify the AWS deployment module locally:
```powershell
& $py -m unittest discover -s autovi_project/aws/tests -p "test_*.py" -v
```
To run all project tests (local + AWS):
```powershell
& $py -m unittest discover -s autovi_project/tests -p "test_*.py" -v
& $py -m unittest discover -s autovi_project/aws/tests -p "test_*.py" -v
```
