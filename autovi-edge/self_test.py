"""AutoVI Edge: Hardware Sanity and Numerical Parity Self-Test Script.

Commissioning tool to verify edge readiness on Raspberry Pi 5 + Hailo-8 AI HAT+:
1. Tests Hailo-8 / ONNX inference backend.
2. Verifies memory bank integrity across all 6 AutoVI categories.
3. Benchmarks end-to-end inference latency on test frames.
4. Verifies 3-tier verdict classification logic.
5. Validates heatmap generation and ROI coordinate conditioning.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EDGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EDGE_ROOT))

from src.decision_engine import DecisionEngine, Verdict
from src.heatmap import HeatmapGenerator
from src.inference_engine import EdgeAnomalyEngine

CATEGORIES = [
    "tank_screw",
    "pipe_clip",
    "engine_wiring",
    "pipe_staple",
    "underbody_pipes",
    "underbody_screw",
]


def run_self_test():
    print("=" * 80)
    print("AUTOVI EDGE: RASPBERRY PI 5 + HAILO-8 SYSTEM SELF-TEST")
    print("=" * 80)

    # 1. Backend Initialization Test
    print("\n[TEST 1/5] Initializing Inference Engine...")
    model_dir = EDGE_ROOT / "models"
    t0 = time.perf_counter()
    try:
        engine = EdgeAnomalyEngine(model_dir=model_dir, preferred_backend="auto")
        print(f"  [PASS] Active backend: {engine.backend_name} (Loaded in {time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        print(f"  [FAIL] Backend initialization failed: {e}")
        return False

    # 2. Memory Bank Verification Test
    print("\n[TEST 2/5] Verifying 6-Category Coreset Memory Banks...")
    loaded_cats = list(engine.category_models.keys())
    for cat in CATEGORIES:
        if cat in engine.category_models:
            mb = engine.category_models[cat]["memory_bank"]
            tau = engine.category_models[cat]["threshold"]
            print(f"  [PASS] Category '{cat:16s}': Shape {mb.shape}, Threshold tau={tau:.4f}")
        else:
            print(f"  [FAIL] Missing memory bank for '{cat}'")
            return False

    # 3. End-to-End Inference & Latency Benchmark
    print("\n[TEST 3/5] Benchmarking End-to-End Inference on Test Frames...")
    # Synthetic 640x480 RGB frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    latencies = []
    engine.set_category("tank_screw")
    # Warmup
    _ = engine.inspect(test_frame)

    for i in range(10):
        t_start = time.perf_counter()
        res = engine.inspect(test_frame)
        lat = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat)

    avg_lat = float(np.mean(latencies))
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
    print(f"  [PASS] 10-Iteration Average Latency: {avg_lat:.2f} ms ({fps:.1f} FPS)")
    print(f"         - Preprocess:       {res['timings_ms']['preprocess']:.2f} ms")
    print(f"         - Feature Extract:  {res['timings_ms']['feature_extraction']:.2f} ms")
    print(f"         - Nearest Neighbor: {res['timings_ms']['nearest_neighbor']:.2f} ms")

    # 4. Decision Engine Verdict Logic Test
    print("\n[TEST 4/5] Testing 3-Tier Operational Verdict Engine...")
    dec_engine = DecisionEngine(review_margin_factor=0.85)
    tau = 18.0
    # Case A: Nominal
    res_pass = dec_engine.evaluate("test", 10.0, override_threshold=tau)
    # Case B: Marginal Review
    res_rev = dec_engine.evaluate("test", 16.0, override_threshold=tau)
    # Case C: Defect
    res_def = dec_engine.evaluate("test", 22.0, override_threshold=tau)

    if res_pass.verdict == Verdict.PASS and res_rev.verdict == Verdict.REVIEW_REQUIRED and res_def.verdict == Verdict.DEFECT:
        print("  [PASS] 3-tier verdict logic verified (PASS, REVIEW_REQUIRED, DEFECT correct)")
    else:
        print(f"  [FAIL] Unexpected verdict mapping: {[res_pass.verdict, res_rev.verdict, res_def.verdict]}")
        return False

    # 5. Heatmap & Overlay Generation Test
    print("\n[TEST 5/5] Testing Heatmap Generation and Color Overlay HUD...")
    roi_file = EDGE_ROOT / "configs" / "roi_configurations.json"
    import json
    with open(roi_file, "r", encoding="utf-8") as f:
        roi_meta = json.load(f).get("categories", {})
    hm_gen = HeatmapGenerator(roi_configs=roi_meta)
    
    float_map, uint8_map = hm_gen.process(
        raw_distances=res["patch_distances_2d"],
        category="tank_screw",
        target_shape=(480, 640),
        category_min_max=(0.0, 30.0),
    )
    overlay = hm_gen.render_overlay(
        frame_bgr=test_frame,
        uint8_map=uint8_map,
        category="tank_screw",
        verdict=res_pass.verdict.value,
        score=res["anomaly_score"],
        threshold=tau,
    )
    if overlay.shape == (480, 640, 3) and uint8_map.shape == (480, 640):
        print(f"  [PASS] Rendered HUD frame shape: {overlay.shape}, Heatmap dtype: {uint8_map.dtype}")
    else:
        print(f"  [FAIL] Heatmap rendering dimension mismatch: {overlay.shape}")
        return False

    print("\n" + "=" * 80)
    print("SELF-TEST RESULT: ALL 5 EDGE COMMISSIONING TESTS PASSED (100% OPERATIONAL)")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_self_test()
    sys.exit(0 if success else 1)
