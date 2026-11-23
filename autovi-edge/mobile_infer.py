#!/usr/bin/env python3
"""AutoVI Edge: Mobile & Embedded Arm Anomaly Inspection Engine.

Supports:
- ONNX Runtime CPU / Mobile backend
- PyTorch FP32 backend
- Single image or live camera stream (OpenCV)
- Full Config C (Deployable Normal ROI + Spatial Coordinate Conditioning + Gaussian Blur)

Usage:
  # Inspect a single image with ONNX:
  python3 mobile_infer.py --category pipe_clip --image path/to/image.jpg

  # Inspect with PyTorch:
  python3 mobile_infer.py --category pipe_clip --image path/to/image.jpg --backend pytorch

  # Run live camera on mobile/laptop:
  python3 mobile_infer.py --category pipe_staple --camera 0
"""

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
ONNX_PATH = MODEL_DIR / "wideresnet50_patchcore.onnx"
PT_PATH = MODEL_DIR / "wideresnet50_patchcore.pt"
MB_DIR = MODEL_DIR / "memory_banks"
CONFIG_FILE = ROOT / "configs" / "roi_configurations.json"

# ImageNet normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MobileAnomalyEngine:
    def __init__(self, backend: str = "onnx", num_threads: int = 4):
        self.backend_type = backend.lower()
        self.num_threads = num_threads

        # Load ROI configurations
        self.roi_configs = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self.roi_configs = json.load(f).get("categories", {})

        # Initialize requested backend
        if self.backend_type == "onnx":
            self._init_onnx()
        elif self.backend_type in ["pytorch", "torch"]:
            self._init_pytorch()
        else:
            raise ValueError(f"Unsupported backend: {backend}. Choose 'onnx' or 'pytorch'.")

        self.banks = {}
        self.active_category = None

    def _init_onnx(self):
        import onnxruntime as ort

        if not ONNX_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_sess = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.ort_sess.get_inputs()[0].name
        self.backend_name = f"ONNX Runtime (Arm CPU, {self.num_threads} threads)"

    def _init_pytorch(self):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        torch.set_num_threads(self.num_threads)

        if PT_PATH.exists():
            print(f"Loading TorchScript model: {PT_PATH.name}...")
            self.pt_model = torch.jit.load(str(PT_PATH), map_location=self.device)
            self.pt_model.eval()
        else:
            print("TorchScript .pt not found; instantiating WideResNet50PatchCore via torchvision...")
            import torchvision.models as models

            class WideResNet50PatchCore(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    bb = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
                    self.conv1, self.bn1, self.relu, self.maxpool = bb.conv1, bb.bn1, bb.relu, bb.maxpool
                    self.layer1, self.layer2, self.layer3 = bb.layer1, bb.layer2, bb.layer3
                    self.avg_pool = torch.nn.AvgPool2d(3, stride=1, padding=1)

                def forward(self, x):
                    x = self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
                    f2 = self.layer2(x)
                    f3 = self.layer3(f2)
                    p2 = self.avg_pool(f2)
                    p3 = self.avg_pool(f3)
                    p3_up = torch.nn.functional.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)
                    return torch.cat([p2, p3_up], dim=1)

            self.pt_model = WideResNet50PatchCore().eval()

        self.backend_name = f"PyTorch FP32 (Arm CPU, {self.num_threads} threads)"

    def set_category(self, category: str):
        if category in self.banks:
            self.active_category = category
            return

        pkl_path = MB_DIR / f"{category}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Memory bank for '{category}' not found at: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        mb = np.asarray(data["memory_bank"], dtype=np.float32)
        mb_sq = np.einsum("ij,ij->i", mb, mb)
        threshold = float(data.get("threshold", 17.50))

        # Setup Config C Deployable ROI and Coordinate Conditioning
        roi_meta = None
        if category in self.roi_configs:
            box = self.roi_configs[category]["deployable_normal_only_roi"]["box"]
            gy1, gx1 = max(0, int(np.floor(box[0] * 32))), max(0, int(np.floor(box[1] * 32)))
            gy2, gx2 = min(32, int(np.ceil(box[2] * 32))), min(32, int(np.ceil(box[3] * 32)))

            cy, cx = (gy1 + gy2) / 2.0, (gx1 + gx2) / 2.0
            sy, sx = max(1.0, (gy2 - gy1) / 2.0), max(1.0, (gx2 - gx1) / 2.0)
            yy, xx = np.mgrid[0:32, 0:32]
            dist_sq = ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2
            spatial_weight = np.exp(-0.2 * dist_sq).astype(np.float32)

            roi_meta = {
                "gy1": gy1, "gy2": gy2, "gx1": gx1, "gx2": gx2,
                "spatial_weight": spatial_weight,
            }

        self.banks[category] = {
            "mb": mb,
            "mb_sq": mb_sq,
            "threshold": threshold,
            "roi": roi_meta,
        }
        self.active_category = category

    def inspect(self, img_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()

        # 1. Preprocess (Resize to 256x256, ImageNet Normalize)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        norm = (img_res - MEAN) / STD

        # 2. Extract Features
        if self.backend_type == "onnx":
            tensor = norm.transpose(2, 0, 1)[None, ...].astype(np.float32)
            out = self.ort_sess.run(None, {self.input_name: tensor})[0]
            # out shape: (1, 1536, 32, 32)
            patches_grid = out[0].transpose(1, 2, 0)
        else:
            with self.torch.inference_mode():
                tensor = self.torch.from_numpy(norm.transpose(2, 0, 1)[None, ...]).float()
                out = self.pt_model(tensor).cpu().numpy()
                patches_grid = out[0].transpose(1, 2, 0)

        # 3. Nearest Neighbor Distance Matching
        cat_info = self.banks[self.active_category]
        mb, mb_sq, roi, tau = cat_info["mb"], cat_info["mb_sq"], cat_info["roi"], cat_info["threshold"]

        if roi is not None:
            gy1, gy2, gx1, gx2 = roi["gy1"], roi["gy2"], roi["gx1"], roi["gx2"]
            roi_patches = patches_grid[gy1:gy2, gx1:gx2].reshape(-1, 1536)

            p_sq = np.sum(roi_patches ** 2, axis=1, keepdims=True)
            dots = np.dot(roi_patches, mb.T)
            d2 = np.maximum(p_sq + mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1))

            roi_dists = min_d.reshape(gy2 - gy1, gx2 - gx1)
            sw = roi["spatial_weight"][gy1:gy2, gx1:gx2]
            weighted_roi = roi_dists * sw

            smoothed_roi = cv2.GaussianBlur(weighted_roi, (5, 5), sigmaX=1.2)
            anomaly_score = float(np.max(smoothed_roi))
        else:
            full_patches = patches_grid.reshape(-1, 1536)
            p_sq = np.sum(full_patches ** 2, axis=1, keepdims=True)
            dots = np.dot(full_patches, mb.T)
            d2 = np.maximum(p_sq + mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1))
            smoothed_2d = cv2.GaussianBlur(min_d.reshape(32, 32), (5, 5), sigmaX=1.2)
            anomaly_score = float(np.max(smoothed_2d))

        is_defect = anomaly_score >= tau
        verdict = "REJECT" if is_defect else "PASS"
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "category": self.active_category,
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(tau, 4),
            "verdict": verdict,
            "is_defect": is_defect,
            "latency_ms": round(total_ms, 1),
            "backend": self.backend_name,
        }


def main():
    parser = argparse.ArgumentParser(description="AutoVI Mobile Inspection Engine")
    parser.add_argument("--category", default="pipe_clip", help="Inspection category")
    parser.add_argument("--image", default="", help="Path to input image file")
    parser.add_argument("--camera", type=int, default=-1, help="Camera index for live feed (e.g. 0)")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "pytorch"], help="Inference backend")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    args = parser.parse_args()

    print("=" * 65)
    print(" AUTOVI EDGE: MOBILE ARMV8/ARMV9 INSPECTION RUNNER")
    print(f" Category: {args.category} | Backend: {args.backend.upper()}")
    print("=" * 65)

    engine = MobileAnomalyEngine(backend=args.backend, num_threads=args.threads)
    engine.set_category(args.category)
    print(f"[ENGINE READY] Active: {engine.backend_name}\n")

    if args.image:
        img_p = Path(args.image)
        if not img_p.exists():
            sys.exit(f"Image not found: {img_p}")
        frame = cv2.imread(str(img_p))
        res = engine.inspect(frame)

        v_color = "\033[91m" if res["is_defect"] else "\033[92m"
        reset = "\033[0m"

        print(f"File:        {img_p.name}")
        print(f"Score:       {res['anomaly_score']:.4f}")
        print(f"Threshold:   {res['threshold']:.4f}")
        print(f"Verdict:     {v_color}{res['verdict']}{reset}")
        print(f"Latency:     {res['latency_ms']:.1f} ms")

    elif args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            sys.exit(f"Failed to open camera device #{args.camera}")
        print(f"Live camera stream active (Camera #{args.camera}). Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = engine.inspect(frame)

            # Overlay HUD
            color = (0, 0, 255) if res["is_defect"] else (0, 255, 0)
            text = f"{res['category'].upper()}: {res['verdict']} | Score: {res['anomaly_score']:.2f} (tau={res['threshold']:.2f}) | {res['latency_ms']:.0f}ms"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("AutoVI Mobile Inspection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Please provide --image <file> or --camera <index> to run inspection.")


if __name__ == "__main__":
    main()
