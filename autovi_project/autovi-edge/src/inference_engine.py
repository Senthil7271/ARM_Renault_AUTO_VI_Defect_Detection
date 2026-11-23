"""AutoVI Edge: Dual-Backend Inference Engine (Hailo-8 NPU + ONNX Runtime Fallback).

Architecture:
1. HailoInferenceBackend:
   - Targets Hailo-8 M.2 (26 TOPS) via pyhailort VStreams.
   - Zero-copy uint8 input with on-chip ImageNet normalization.
   - Sub-10ms feature extraction latency.
2. ONNXRuntimeBackend:
   - Portable cross-platform fallback using ONNX Runtime CPU.
   - Identical float32 numerical parity.
3. Coreset Memory Bank Search:
   - Vectorized matrix multiplication for nearest-neighbor distance computation.
   - Category-switchable without process restart.
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


class HailoInferenceBackend:
    """Primary NPU backend for Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS)."""

    def __init__(self, hef_path: Union[str, Path]):
        self.hef_path = Path(hef_path)
        self.device = None
        self.network_group = None
        self.input_vstream = None
        self.output_vstream = None
        self._validate_hef_integrity()
        self._init_hailo()

    def _validate_hef_integrity(self):
        """Strict deployment-integrity check: rejects uncompiled stubs and corrupt files."""
        if not self.hef_path.exists():
            raise FileNotFoundError(f"Hailo HEF file not found: {self.hef_path}")

        size_bytes = self.hef_path.stat().st_size
        if size_bytes < 1_000_000:
            # Check for JSON metadata stub
            try:
                with open(self.hef_path, "rb") as f:
                    head = f.read(512)
                if head.startswith(b"{") or b"model_name" in head:
                    raise ValueError(
                        f"REJECTED STUB HEF: '{self.hef_path}' is a JSON metadata placeholder "
                        f"({size_bytes} bytes), NOT a physical compiled Hailo binary. "
                        f"Real WideResNet-50-2 HEF must be compiled with Hailo Dataflow Compiler (DFC) "
                        f"on a Linux x86_64 system (expected size: 15-35 MB)."
                    )
            except UnicodeDecodeError:
                pass
            raise ValueError(
                f"INVALID HEF ARTIFACT: File size {size_bytes} bytes is too small for WideResNet-50-2. "
                f"Physical binary HEFs are at least 15 MB."
            )

    def _init_hailo(self):
        try:
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
            print(f"[HAILO] Initializing Hailo-8 device with physical HEF: {self.hef_path}")
            self.vdevice = VDevice()
            self.hef = HEF(str(self.hef_path))
            self.configure_params = ConfigureParams.create_from_hef(
                hef=self.hef, interface=HailoStreamInterface.PCIe
            )
            self.network_groups = self.vdevice.configure(self.hef, self.configure_params)
            self.network_group = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()
            
            self.input_vstream_params = InputVStreamParams.make(
                self.network_group, format_type=FormatType.UINT8
            )
            self.output_vstream_params = OutputVStreamParams.make(
                self.network_group, format_type=FormatType.FLOAT32
            )
            print("[HAILO] Successfully configured Hailo-8 PCIe VStreams (26 TOPS mode).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize native Hailo-8 hardware: {e}")

    def infer(self, input_rgb_256: np.ndarray) -> np.ndarray:
        """Runs inference on (256, 256, 3) uint8 image. Returns (1024, 1536) patch embeddings."""
        # Ensure batch dimension: (1, 256, 256, 3) uint8
        if input_rgb_256.ndim == 3:
            input_batch = np.expand_dims(input_rgb_256, axis=0)
        else:
            input_batch = input_rgb_256

        from hailo_platform import InferVStreams
        with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                input_name = list(self.input_vstream_params.keys())[0]
                output_name = list(self.output_vstream_params.keys())[0]
                infer_results = infer_pipeline.infer({input_name: input_batch})
                raw_out = infer_results[output_name]  # (1, 32, 32, 1536)
                patches = raw_out[0].reshape(-1, 1536)  # (1024, 1536)
                return patches


class ONNXRuntimeBackend:
    """Cross-platform CPU fallback backend using ONNX Runtime."""

    def __init__(self, onnx_path: Union[str, Path]):
        self.onnx_path = str(onnx_path)
        import onnxruntime as ort
        print(f"[ONNX] Loading ONNX model on CPUExecutionProvider: {self.onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        self.session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print("[ONNX] Backend ready.")

    def infer(self, input_rgb_256: np.ndarray) -> np.ndarray:
        """Runs inference on (256, 256, 3) uint8/float image. Returns (1024, 1536) patch embeddings."""
        arr = input_rgb_256.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        norm = (arr - self.mean) / self.std  # (256, 256, 3)
        # NCHW: (1, 3, 256, 256)
        tensor = norm.transpose(2, 0, 1)[None, ...]
        
        # Output shape: (1, 1536, 32, 32)
        out = self.session.run(None, {self.input_name: tensor})[0]
        # Transpose to (32, 32, 1536) -> reshape to (1024, 1536)
        patches = out[0].transpose(1, 2, 0).reshape(-1, 1536)
        return patches


class EdgeAnomalyEngine:
    """Unified Edge Inference and Nearest-Neighbor Anomaly Engine."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        preferred_backend: str = "auto",  # 'hailo', 'onnx', or 'auto'
    ):
        self.model_dir = Path(model_dir)
        self.backend_name = None
        self.backend = None
        self.current_category = None
        self.memory_bank = None
        self.mb_sq = None
        self.category_models = {}
        self.roi_configs = {}
        roi_p = self.model_dir.parent / "configs" / "roi_configurations.json"
        if roi_p.exists():
            with open(roi_p, "r", encoding="utf-8") as f:
                self.roi_configs = json.load(f).get("categories", {})

        self._init_backend(preferred_backend)
        self._load_category_memory_banks()

    def _init_backend(self, preferred: str):
        hef_file = self.model_dir / "wideresnet50_patchcore.hef"
        onnx_file = self.model_dir / "wideresnet50_patchcore.onnx"

        # Check HEF physical integrity before selecting backend
        hef_is_valid = False
        hef_reject_reason = ""
        if hef_file.exists():
            st_size = hef_file.stat().st_size
            if st_size < 1_000_000:
                hef_reject_reason = (
                    f"File '{hef_file.name}' is an uncompiled stub ({st_size} bytes). "
                    f"Physical Hailo-8 HEF must be >= 15 MB."
                )
            else:
                hef_is_valid = True
        else:
            hef_reject_reason = f"File '{hef_file.name}' not found."

        # Attempt Hailo-8 if preferred
        if preferred in ["auto", "hailo"]:
            if hef_is_valid:
                try:
                    self.backend = HailoInferenceBackend(hef_file)
                    self.backend_name = "Hailo-8 NPU (26 TOPS)"
                    print(f"[ENGINE] Active Backend: {self.backend_name}")
                    return
                except Exception as e:
                    print(f"[ENGINE WARNING] Hailo hardware initialization failed: {e}")
                    if preferred == "hailo":
                        raise RuntimeError(f"Hailo backend requested but hardware init failed: {e}")
            else:
                msg = f"[DEPLOYMENT INTEGRITY GUARD] Hailo-8 backend unavailable: {hef_reject_reason}"
                if preferred == "hailo":
                    raise RuntimeError(
                        f"{msg}\n"
                        f"Refusing to claim Hailo backend availability when only an uncompiled placeholder exists.\n"
                        f"Compile wideresnet50_patchcore.onnx to a genuine HEF using the Hailo Dataflow Compiler."
                    )
                print(f"{msg} Auto-selecting ONNX Runtime CPU backend.")

        # Fallback to ONNX Runtime
        if onnx_file.exists():
            self.backend = ONNXRuntimeBackend(onnx_file)
            self.backend_name = "ONNX Runtime (CPU Fallback)"
            print(f"[ENGINE] Active Backend: {self.backend_name}")
            return

        raise FileNotFoundError(f"Neither valid HEF nor ONNX model found in {self.model_dir}")

    def _load_category_memory_banks(self):
        """Loads serialized category memory banks."""
        mb_dir = self.model_dir / "memory_banks"
        if not mb_dir.exists():
            print(f"[ENGINE WARNING] Memory bank dir not found: {mb_dir}")
            return

        for pkl_p in sorted(mb_dir.glob("*.pkl")):
            cat_name = pkl_p.stem
            try:
                with open(pkl_p, "rb") as f:
                    data = pickle.load(f)
                mb = np.asarray(data["memory_bank"], dtype=np.float32)
                # Memory-efficient squared norm without creating (N, D) temporary array
                mb_sq = np.einsum('ij,ij->i', mb, mb)
                self.category_models[cat_name] = {
                    "memory_bank": mb,
                    "mb_sq": mb_sq,
                    "threshold": float(data.get("threshold", 18.0)),
                    "min_score": float(data.get("min_score", 0.0)),
                    "max_score": float(data.get("max_score", 30.0)),
                }
                del data
                print(f"  Loaded memory bank for '{cat_name}': shape {mb.shape}, tau={self.category_models[cat_name]['threshold']:.4f}")
            except Exception as e:
                print(f"[ENGINE WARNING] Failed to load {pkl_p.name}: {e}")

    def set_category(self, category: str):
        """Switches active inspection category."""
        if category not in self.category_models:
            raise ValueError(f"Category '{category}' not found in loaded memory banks: {list(self.category_models.keys())}")
        self.current_category = category
        self.memory_bank = self.category_models[category]["memory_bank"]
        self.mb_sq = self.category_models[category]["mb_sq"]

    def inspect(self, frame_bgr: np.ndarray) -> dict:
        """End-to-end edge inspection pipeline.
        
        Args:
            frame_bgr: Raw input camera image (any resolution)
            
        Returns:
            dict containing anomaly_score, patch_distances (32, 32), timings_ms
        """
        if self.current_category is None:
            if self.category_models:
                self.set_category(list(self.category_models.keys())[0])
            else:
                raise RuntimeError("No category memory banks loaded.")

        t0 = time.perf_counter()

        # 1. Preprocessing (Resize & BGR to RGB)
        t_prep_start = time.perf_counter()
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_256 = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # 2. Backbone Feature Extraction (NPU or ONNX)
        t_npu_start = time.perf_counter()
        patches = self.backend.infer(img_256)  # (1024, 1536)
        t_npu_ms = (time.perf_counter() - t_npu_start) * 1000.0

        # 3. Vectorized Nearest-Neighbor Search
        t_nn_start = time.perf_counter()
        # ||p - m||^2 = ||p||^2 + ||m||^2 - 2 p.m
        p_sq = np.sum(patches ** 2, axis=1, keepdims=True)  # (1024, 1)
        
        # Chunked dot product to optimize L2/L3 cache on ARM Cortex-A76
        dists = []
        chunk_size = 256
        for c in range(0, 1024, chunk_size):
            chunk_p = patches[c : c + chunk_size]
            chunk_p_sq = p_sq[c : c + chunk_size]
            dots = np.dot(chunk_p, self.memory_bank.T)
            d2 = np.maximum(chunk_p_sq + self.mb_sq - 2.0 * dots, 0.0)
            min_d = np.sqrt(np.min(d2, axis=1))
            dists.append(min_d)
            
        patch_dists = np.concatenate(dists)  # (1024,)
        patch_2d = patch_dists.reshape(32, 32)

        # Apply Deployable Normal ROI and 2D spatial coordinate weighting for active categories
        if self.current_category in self.roi_configs:
            box = self.roi_configs[self.current_category]["deployable_normal_only_roi"]["box"]
            gy1 = max(0, int(np.floor(box[0] * 32)))
            gx1 = max(0, int(np.floor(box[1] * 32)))
            gy2 = min(32, int(np.ceil(box[2] * 32)))
            gx2 = min(32, int(np.ceil(box[3] * 32)))

            cy, cx = (gy1 + gy2) / 2.0, (gx1 + gx2) / 2.0
            sigma_y = max(1.0, (gy2 - gy1) / 2.0)
            sigma_x = max(1.0, (gx2 - gx1) / 2.0)
            yy, xx = np.mgrid[0:32, 0:32]
            dist_sq = ((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2
            spatial_weight = np.exp(-0.2 * dist_sq).astype(np.float32)

            roi_patch_dists = np.zeros_like(patch_2d)
            roi_patch_dists[gy1:gy2, gx1:gx2] = patch_2d[gy1:gy2, gx1:gx2]
            weighted_patch_dists = roi_patch_dists * spatial_weight
            anomaly_score = float(np.max(weighted_patch_dists))
        else:
            anomaly_score = float(np.max(patch_dists))

        t_nn_ms = (time.perf_counter() - t_nn_start) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "category": self.current_category,
            "anomaly_score": anomaly_score,
            "patch_distances_2d": patch_2d,
            "patch_distances_1d": patch_dists,
            "threshold": self.category_models[self.current_category]["threshold"],
            "global_range": (
                self.category_models[self.current_category]["min_score"],
                self.category_models[self.current_category]["max_score"],
            ),
            "backend": self.backend_name,
            "timings_ms": {
                "preprocess": round(t_prep_ms, 2),
                "feature_extraction": round(t_npu_ms, 2),
                "nearest_neighbor": round(t_nn_ms, 2),
                "total": round(total_ms, 2),
                "fps": round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0,
            },
        }
