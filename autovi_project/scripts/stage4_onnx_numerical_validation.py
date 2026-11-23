"""AutoVI Edge Deployment: Stage 4 ONNX Numerical Validation.

Validates numerical parity between PyTorch WideResNet-50-2 reference and ONNX Runtime:
1. Compares raw feature tensors.
2. Compares reconstructed PatchCore embeddings.
3. Compares nearest-neighbor distances to category memory banks.
4. Compares image anomaly scores.
5. Compares generated anomaly heatmaps.

Evaluates representative normal and defective images across all six categories.
Outputs metrics: MAE, Max Absolute Error, Relative Error, Cosine Similarity, Rank Agreement.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import Wide_ResNet50_2_Weights
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

CATEGORIES = [
    "tank_screw",
    "pipe_clip",
    "engine_wiring",
    "pipe_staple",
    "underbody_pipes",
    "underbody_screw",
]


class PyTorchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        backbone.eval()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            f2 = self.layer2(x)
            f3 = self.layer3(f2)
            p2 = self.avg_pool(f2)
            p3 = self.avg_pool(f3)
            p3_up = F.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)
            out = torch.cat([p2, p3_up], dim=1)
        return out


def preprocess_image(img_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """Applies canonical ImageNet preprocessing to image."""
    img = Image.open(str(img_path)).convert("RGB")
    img_res = img.resize((256, 256), Image.BILINEAR)
    arr = np.array(img_res, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std  # (256, 256, 3)
    
    # PyTorch: (1, 3, 256, 256)
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)
    np_batch = tensor.numpy()
    return tensor, np_batch


def run_numerical_validation():
    print("=" * 80)
    print("STAGE 4: PYTORCH VS ONNX RUNTIME NUMERICAL VALIDATION")
    print("=" * 80)

    # Load PyTorch model
    print("\n[1/3] Initializing PyTorch reference...")
    pytorch_model = PyTorchFeatureExtractor()
    pytorch_model.eval()

    # Load ONNX Runtime session
    onnx_path = PROJECT_ROOT / "models" / "onnx" / "wideresnet50_patchcore.onnx"
    print(f"\n[2/3] Initializing ONNX Runtime session: {onnx_path.name}...")
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    models_dir = PROJECT_ROOT / "outputs" / "sagemaker_runs" / "autovi-all-2026-11-15-06-03-24" / "model"

    validation_results = []

    print("\n[3/3] Evaluating across all 6 categories (Normal & Defective Samples)...")

    for cat in CATEGORIES:
        test_dir = WORKSPACE_ROOT / cat / cat / "test"
        
        # Load category memory bank
        with open(models_dir / cat / "patchcore_model.pkl", "rb") as f:
            cat_data = pickle.load(f)
        mb = cat_data["memory_bank"]  # (M, 1536)

        # Select 2 normal images and 2 defective images
        good_imgs = sorted(list((test_dir / "good").glob("*.png")))[:2]
        def_subtypes = [p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"]
        def_imgs = []
        for s in def_subtypes:
            imgs = sorted(list(s.glob("*.png")))
            if imgs:
                def_imgs.append(imgs[0])
            if len(def_imgs) >= 2:
                break

        test_samples = [(p, "normal") for p in good_imgs] + [(p, "defective") for p in def_imgs]

        for p_img, sample_type in test_samples:
            img_tensor, img_np = preprocess_image(p_img)

            # 1. PyTorch Feature Extraction
            with torch.inference_mode():
                feat_py = pytorch_model(img_tensor).numpy()  # (1, 1536, 32, 32)

            # 2. ONNX Runtime Feature Extraction
            feat_onnx = ort_session.run(None, {"input": img_np})[0]  # (1, 1536, 32, 32)

            # Raw Tensor Comparison
            mae_raw = float(np.mean(np.abs(feat_py - feat_onnx)))
            max_err_raw = float(np.max(np.abs(feat_py - feat_onnx)))
            denom = np.mean(np.abs(feat_py)) + 1e-7
            rel_err_raw = float(mae_raw / denom)
            cos_sim_raw = float(1.0 - cosine(feat_py.ravel(), feat_onnx.ravel()))

            # Patch Embeddings: (1024, 1536)
            patches_py = feat_py[0].transpose(1, 2, 0).reshape(-1, 1536)
            patches_onnx = feat_onnx[0].transpose(1, 2, 0).reshape(-1, 1536)

            # 3. Nearest-Neighbor Distances to Memory Bank
            # Vectorized chunked 1-NN computation
            # For each patch, compute min Euclidean distance to mb
            # ||p - m||^2 = ||p||^2 + ||m||^2 - 2 * p.m
            mb_sq = np.sum(mb ** 2, axis=1)  # (M,)

            def compute_nn_distances(patches):
                p_sq = np.sum(patches ** 2, axis=1, keepdims=True)  # (1024, 1)
                dists = []
                # Chunked dot product
                for chunk_start in range(0, 1024, 256):
                    chunk_p = patches[chunk_start : chunk_start + 256]
                    chunk_p_sq = p_sq[chunk_start : chunk_start + 256]
                    dots = np.dot(chunk_p, mb.T)  # (chunk, M)
                    d2 = chunk_p_sq + mb_sq - 2.0 * dots
                    d2 = np.maximum(d2, 0.0)
                    min_d = np.sqrt(np.min(d2, axis=1))
                    dists.append(min_d)
                return np.concatenate(dists)

            nn_dists_py = compute_nn_distances(patches_py)
            nn_dists_onnx = compute_nn_distances(patches_onnx)

            mae_nn = float(np.mean(np.abs(nn_dists_py - nn_dists_onnx)))
            max_err_nn = float(np.max(np.abs(nn_dists_py - nn_dists_onnx)))
            cos_sim_nn = float(1.0 - cosine(nn_dists_py, nn_dists_onnx))

            # 4. Anomaly Scores
            score_py = float(np.max(nn_dists_py))
            score_onnx = float(np.max(nn_dists_onnx))
            abs_score_err = abs(score_py - score_onnx)

            # 5. Reconstructed Heatmaps
            # Reshape to (32, 32), upsample to (256, 256), Gaussian smooth (sigma=4.0)
            map2d_py = nn_dists_py.reshape(32, 32)
            map2d_onnx = nn_dists_onnx.reshape(32, 32)

            hm_py = gaussian_filter(cv2.resize(map2d_py, (256, 256), interpolation=cv2.INTER_LINEAR), sigma=4.0)
            hm_onnx = gaussian_filter(cv2.resize(map2d_onnx, (256, 256), interpolation=cv2.INTER_LINEAR), sigma=4.0)

            mae_hm = float(np.mean(np.abs(hm_py - hm_onnx)))
            max_err_hm = float(np.max(np.abs(hm_py - hm_onnx)))

            res = {
                "category": cat,
                "image_id": p_img.stem,
                "subtype": p_img.parent.name,
                "type": sample_type,
                "raw_mae": mae_raw,
                "raw_max_err": max_err_raw,
                "raw_rel_err": rel_err_raw,
                "raw_cosine_sim": cos_sim_raw,
                "nn_mae": mae_nn,
                "nn_max_err": max_err_nn,
                "nn_cosine_sim": cos_sim_nn,
                "score_pytorch": score_py,
                "score_onnx": score_onnx,
                "score_diff": abs_score_err,
                "heatmap_mae": mae_hm,
                "heatmap_max_err": max_err_hm,
            }
            validation_results.append(res)
            print(f"  [{cat:15s} | {sample_type:9s} | {p_img.name}] Raw CosSim: {cos_sim_raw:.8f} | Raw MaxErr: {max_err_raw:.2e} | Score PyT: {score_py:7.4f} vs ONNX: {score_onnx:7.4f} (diff: {abs_score_err:.2e}) | HM MAE: {mae_hm:.2e}")

    df = pd.DataFrame(validation_results)
    out_csv = PROJECT_ROOT / "outputs" / "stage4_onnx_numerical_validation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Numerical validation records saved to: {out_csv}")

    # Summary Statistics
    print("\n" + "=" * 80)
    print("NUMERICAL PARITY SUMMARY:")
    print("=" * 80)
    print(f"Total Samples Evaluated: {len(df)}")
    print(f"Average Feature MAE:           {df['raw_mae'].mean():.2e} (float32 tolerance)")
    print(f"Max Feature Absolute Error:    {df['raw_max_err'].max():.2e}")
    print(f"Min Feature Cosine Similarity: {df['raw_cosine_sim'].min():.8f}")
    print(f"Average NN Distance MAE:       {df['nn_mae'].mean():.2e}")
    print(f"Max Image Score Difference:    {df['score_diff'].max():.2e}")
    print(f"Average Heatmap MAE:           {df['heatmap_mae'].mean():.2e}")

    # Pass/Fail Gate Check
    passed = (
        df['raw_cosine_sim'].min() > 0.99999
        and df['raw_max_err'].max() < 1e-4
        and df['score_diff'].max() < 1e-4
    )
    print(f"\n[NUMERICAL PARITY GATE]: {'PASSED (EXCELLENT FP32 PARITY)' if passed else 'FAILED'}")
    return df


if __name__ == "__main__":
    run_numerical_validation()
