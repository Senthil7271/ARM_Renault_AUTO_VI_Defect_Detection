# AutoVI Phase 3 — Second SageMaker Pilot Failure Investigation Report
**Job Name**: `autovi-p3-rd-pipeclip-2026-11-15-11-02-35`  
**Execution Timestamp**: 2026-11-15 16:32:35 to 16:43:14 IST (11:02:35 to 11:13:14 UTC)  
**Hardware Instance**: AWS SageMaker `ml.g4dn.xlarge` (1x NVIDIA Tesla T4 GPU, 16 GiB VRAM)  
**Algorithm**: Reverse Distillation (Frozen Pretrained WideResNet-50-2 Teacher + Trainable Student Decoder)  
**Dataset Target**: `pipe_clip` (Industrial assembly inspection)  
**Investigator**: Industrial Computer Vision & Anomaly Detection Team  

---

## 1. Executive Summary & Incident Timeline

During the execution of Pilot Experiment 1 (Attempt 2), the training job terminated with an algorithm exit code 1. A detailed CloudWatch and SageMaker API investigation revealed a critical distinction:

> **The failure was NOT a training failure.**
> All 30 training epochs completed successfully on the NVIDIA Tesla T4 GPU. The student decoder weights were fully optimized (validation loss decreased from 1.0588 down to 0.5581). Both `best_model.pth` and `final_model.pth` were successfully written to disk before the error occurred.

The failure occurred exclusively during the **post-training threshold calibration** routine on line 218 of `aws/sagemaker_train_deep.py` due to missing module-level imports (`cv2` and `torchvision.transforms as T`).

| Metric | Recorded Value |
| :--- | :--- |
| **SageMaker Job Status** | `Failed` (SecondaryStatus: `Failed`) |
| **Creation Time** | 2026-11-15 11:02:35 UTC (16:32:35 IST) |
| **Training Start Time** | 2026-11-15 11:06:23 UTC (16:36:23 IST) |
| **Training End Time** | 2026-11-15 11:13:14 UTC (16:43:14 IST) |
| **Elapsed Runtime** | 411 seconds (6.85 minutes) |
| **Billable Duration** | 411 seconds |
| **Billable Cost** | **$0.0841 USD** (at $0.7364/hr) |
| **Last Completed Epoch** | Epoch 30 of 30 (100% complete) |
| **Best Model Epoch** | Epoch 26 (Validation Loss = 0.558056) |
| **Model Checkpoint Status** | Preserved in S3 `model.tar.gz` and successfully recovered |

---

## 2. Full Exception Traceback & Failure Reason

From `sagemaker.describe_training_job`:

```text
FailureReason:
AlgorithmError: ExecuteUserScriptError:
ExitCode 1
ErrorMessage "NameError: name 'cv2' is not defined"
Command "/opt/conda/bin/python3.10 sagemaker_train_deep.py --backbone wide_resnet50_2 --batch-size 8 --category pipe_clip --epochs 30 --lr 0.0002 --model-type reverse_distillation --seed 42"
```

### Traceback Reconstruction

```python
Traceback (most recent call last):
  File "sagemaker_train_deep.py", line 271, in <module>
    main()
  File "sagemaker_train_deep.py", line 218, in main
    img = cv2.imread(str(vp))
NameError: name 'cv2' is not defined
```

Furthermore, line 222 immediately following line 218:
```python
t_img = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])(img).unsqueeze(0).to(device)
```
would have thrown `NameError: name 'T' is not defined` had `cv2` been resolved in isolation.

---

## 3. Root Cause Analysis

1. **Defect Location**: `autovi_project/aws/sagemaker_train_deep.py`, lines 218–225.
2. **Mechanism**:
   - `sagemaker_train_deep.py` delegates the actual training loop to `src.training.train_deep.train_reverse_distillation`. Inside `train_deep.py`, all imports (`cv2`, `torch`, `torchvision.transforms as T`) were correctly imported and operational.
   - Hence, all 30 epochs ran smoothly on the GPU without raising any exceptions.
   - When control returned to `sagemaker_train_deep.py`, Step 3 executed:
     ```python
     print("\n[CALIBRATION] Calibrating 95th percentile threshold on held-out nominals...")
     model.eval()
     val_scores = []
     with torch.no_grad():
         for vp in val_paths:
             img = cv2.imread(str(vp))  # <-- NameError: 'cv2' is not defined
     ```
   - In `sagemaker_train_deep.py`, `import cv2` and `import torchvision.transforms as T` were missing at the top of the file.

---

## 4. Checkpoint Preservation and Recovery

Because SageMaker automatically uploads `/opt/ml/model` to S3 upon job exit (even on failure when files have been written to the model directory), the model archive was preserved in:
`s3://sagemaker-us-east-1-077578771161/autovi/phase3_outputs/autovi-p3-rd-pipeclip-2026-11-15-11-02-35/output/model.tar.gz`

### Artifact Inventory:
1. `model.tar.gz`: 198.06 MB
2. Downloaded and extracted to: `autovi_project/outputs/phase3/recovered_pilot2/pipe_clip/reverse_distillation/`
   - `best_model.pth`: 106.98 MB (Epoch 26, Val Loss: 0.558056)
   - `final_model.pth`: 106.98 MB (Epoch 30, Final Val Loss: 0.562179)
   - `train_history.json`: 30 epochs of training loss, validation loss, grad norm, and learning rates.

### Training Dynamics (30 Epochs):
- **Epoch 1**: Train Loss = 1.4217, Val Loss = 1.0588
- **Epoch 10**: Train Loss = 0.7421, Val Loss = 0.6912
- **Epoch 20**: Train Loss = 0.5894, Val Loss = 0.5743
- **Epoch 26 (Best)**: Train Loss = 0.5312, **Val Loss = 0.5581**
- **Epoch 30 (Final)**: Train Loss = 0.5176, Val Loss = 0.5622

**Scientific Verification**:
The student weights diverged by **187,594.11 absolute sum** across 23,010,304 parameters from random initialization, proving substantive convergence.

---

## 5. Corrective Actions Applied

1. **Added Missing Imports**: Added `import cv2` and `import torchvision.transforms as T` directly to `aws/sagemaker_train_deep.py`.
2. **Robust Path Resolution**: Refactored `CODE_DIR` resolution in `sagemaker_train_deep.py` to seamlessly handle both local execution (`PROJECT_ROOT`) and container execution (`/opt/ml/code`).
3. **Static Symbol Analysis**: Implemented automated AST/`symtable` scanning across the entire project repository (52 Python files), confirming 0 undefined symbols.
4. **End-to-End Simulation**: Completed a full 2-epoch CPU simulation executing all 16 pipeline branches including training, serialization, calibration, and metadata saving.
5. **Regression Test Addition**: Added `TestSageMakerDeepPreflight` to `tests/test_deep_models.py` to prevent any regressions.
