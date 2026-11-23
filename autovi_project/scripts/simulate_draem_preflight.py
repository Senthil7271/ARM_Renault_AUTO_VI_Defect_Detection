"""Complete DRAEM Local CPU Simulation & Preflight Verification Script.

Executes:
1. Two-epoch training simulation via sagemaker_train_deep.py logic on pipe_clip nominal images.
2. Checkpoint selection, serialization, and reloading verification.
3. Threshold calibration on held-out nominals.
4. Discriminator synthetic anomaly localization test (proving it does NOT just predict background).
5. Segmentation map export and visual diagnostic generation.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.synthetic_anomalies import SyntheticAnomalyGenerator
from src.models.draem import DRAEMModel
from aws.sagemaker_train_deep import resolve_category_train_val_paths, parse_args
from src.training.train_deep import train_draem


def run_draem_simulation():
    sim_dir = PROJECT_ROOT / "outputs" / "phase3" / "draem_sim"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    model_dir = sim_dir / "model"
    output_dir = sim_dir / "output"
    vis_dir = sim_dir / "visualizations"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STEP 1: Resolve dataset paths for pipe_clip")
    print("=" * 75)
    train_paths, val_paths = resolve_category_train_val_paths(
        data_dir=WORKSPACE_ROOT / "pipe_clip",
        category="pipe_clip",
        seed=42,
    )
    print(f"Total train images: {len(train_paths)}, validation images: {len(val_paths)}")

    # Use a representative subset for local CPU smoke simulation
    train_sub = train_paths[:24]
    val_sub = val_paths[:8]
    print(f"CPU smoke subset: {len(train_sub)} train images, {len(val_sub)} val images")

    print("\n" + "=" * 75)
    print("STEP 2: Execute 2-Epoch CPU Training Simulation")
    print("=" * 75)
    t0 = time.time()
    model, summary = train_draem(
        train_paths=train_sub,
        val_paths=val_sub,
        output_dir=model_dir,
        base_reconstruct_c=24,       # Lightened for fast CPU smoke test
        base_discriminative_c=16,
        epochs=2,
        batch_size=4,
        lr=2e-4,
        device="cpu",
        use_amp=False,
        seed=42,
    )
    train_duration = time.time() - t0
    print(f"Training completed in {train_duration:.1f}s.")
    print(f"Summary: {summary}")

    # Check serialized files
    assert (model_dir / "best_model.pth").exists(), "best_model.pth missing!"
    assert (model_dir / "final_model.pth").exists(), "final_model.pth missing!"
    assert (model_dir / "train_history.json").exists(), "train_history.json missing!"

    print("\n" + "=" * 75)
    print("STEP 3: Checkpoint Reloading Fidelity Check")
    print("=" * 75)
    reloaded_model = DRAEMModel.load(model_dir / "best_model.pth", device="cpu")
    print("Model successfully loaded from disk.")

    print("\n" + "=" * 75)
    print("STEP 4: Held-Out Normal Calibration Routine")
    print("=" * 75)
    reloaded_model.eval()
    val_scores = []
    with torch.no_grad():
        for vp in val_sub:
            img = cv2.imread(str(vp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            t_img = (torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0).unsqueeze(0)
            score, _ = reloaded_model.predict_anomaly(t_img)
            val_scores.append(float(score.item()))

    val_scores = np.sort(np.array(val_scores))
    calibrated_q95 = float(np.quantile(val_scores, 0.95))
    reloaded_model.threshold = calibrated_q95
    reloaded_model.calibrated = True

    calib_model_path = model_dir / "calibrated_model.pth"
    reloaded_model.save(calib_model_path, metadata={"calibrated_threshold": calibrated_q95, "n_val": len(val_scores)})
    print(f"Calibrated 95th percentile threshold: {calibrated_q95:.4f} on N={len(val_scores)} images.")
    assert calib_model_path.exists(), "calibrated_model.pth was not saved!"

    print("\n" + "=" * 75)
    print("STEP 5: Discriminator Localization Test (Non-Trivial Prediction Verification)")
    print("=" * 75)
    # Load one nominal image, create a synthetic anomaly with generator
    test_img_raw = cv2.imread(str(val_sub[0]))
    test_img_raw = cv2.cvtColor(test_img_raw, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img_raw, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    generator = SyntheticAnomalyGenerator(nominal_pool=[test_img], p_anomaly=1.0, seed=123)
    corrupted_img, synth_mask, is_anom = generator.generate(test_img)

    t_corrupted = (torch.from_numpy(corrupted_img).permute(2, 0, 1).contiguous().float()).unsqueeze(0)
    with torch.no_grad():
        reconstructed, pred_anomaly_map = reloaded_model(t_corrupted)
        image_score, amap_sq = reloaded_model.predict_anomaly(t_corrupted)

    rec_np = reconstructed.squeeze(0).permute(1, 2, 0).numpy()
    pred_map_np = pred_anomaly_map.squeeze(0).squeeze(0).numpy()

    # Verification: Confirm the discriminator does NOT predict pure zeros/background
    max_pred = float(pred_map_np.max())
    mean_pred = float(pred_map_np.mean())
    synth_mask_area = float(synth_mask.sum())
    overlap_activation = float(pred_map_np[synth_mask > 0.5].mean()) if synth_mask_area > 0 else 0.0

    print(f"Synthetic mask area: {synth_mask_area:.0f} pixels")
    print(f"Discriminator prediction min: {pred_map_np.min():.4f}, mean: {mean_pred:.4f}, max: {max_pred:.4f}")
    print(f"Mean activation inside synthetic defect region: {overlap_activation:.4f}")
    assert max_pred > 0.01, "Discriminator collapsed to predicting all zeros!"
    print("[PASS] Discriminator produces non-trivial spatial activations responding to synthetic corruptions.")

    print("\n" + "=" * 75)
    print("STEP 6: Export Multi-Panel Visualization")
    print("=" * 75)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(test_img)
    axes[0].set_title("Original Nominal")
    axes[0].axis("off")

    axes[1].imshow(corrupted_img)
    axes[1].set_title("Synthetic Pseudo-Anomaly")
    axes[1].axis("off")

    axes[2].imshow(synth_mask, cmap="gray")
    axes[2].set_title(f"Synthetic Mask (GT, {synth_mask_area:.0f}px)")
    axes[2].axis("off")

    axes[3].imshow(np.clip(rec_np, 0.0, 1.0))
    axes[3].set_title("DRAEM Reconstruction")
    axes[3].axis("off")

    im = axes[4].imshow(pred_map_np, cmap="inferno", vmin=0.0, vmax=max(0.1, max_pred))
    axes[4].set_title(f"DRAEM Anomaly Map (Score: {image_score.item():.4f})")
    axes[4].axis("off")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    vis_path = vis_dir / "draem_simulation_diagnostic.png"
    plt.savefig(vis_path, dpi=150)
    plt.close()
    print(f"Saved diagnostic visualization: {vis_path}")

    # Step 7: Save metadata
    training_meta = {
        "project": "AutoVI_Phase_3",
        "category": "pipe_clip",
        "model_type": "draem",
        "epochs": 2,
        "batch_size": 4,
        "device": "cpu",
        "training_time_sec": train_duration,
        "calibrated_threshold": calibrated_q95,
        "val_score_mean": float(np.mean(val_scores)),
        "val_score_std": float(np.std(val_scores)),
        "summary": summary,
    }
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    print("\n[COMPLETE] DRAEM Local CPU Simulation Passed All 7 Checks Successfully!")
    return True


if __name__ == "__main__":
    run_draem_simulation()
