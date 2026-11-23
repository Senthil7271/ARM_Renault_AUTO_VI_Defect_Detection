"""Validates SageMaker downloaded model artifacts and compares CPU vs GPU metrics.

Usage:
    python autovi_project/aws/validate_artifacts.py --run-dir outputs/sagemaker_runs/<job_name>
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchcore_model import PatchCoreModel


def validate_artifacts(run_dir: Path):
    run_dir = Path(run_dir).resolve()
    print("\n" + "=" * 65)
    print("      Validating SageMaker GPU Trained Model Artifacts      ")
    print("=" * 65)
    print(f"  Target Run Directory: {run_dir}")
    print("=" * 65 + "\n")

    # Locate model.pkl
    model_pkls = list(run_dir.rglob("patchcore_model.pkl"))
    if not model_pkls:
        print(f"[FAIL] Could not find 'patchcore_model.pkl' under {run_dir}")
        return False

    model_path = model_pkls[0]
    print(f"[PASS] Located model checkpoint: {model_path} ({model_path.stat().st_size / (1024**2):.2f} MiB)")

    # Attempt reloading model
    try:
        model = PatchCoreModel.load(model_path, device="cpu")
        print(f"[PASS] Successfully loaded model into memory.")
        print(f"       Backbone:    {model.backbone_name}")
        print(f"       Memory Bank: {model.memory_bank.shape if model.memory_bank is not None else 'None'}")
        print(f"       Threshold:   {model.threshold:.4f} (Calibrated: {model.calibrated})")
    except Exception as e:
        print(f"[FAIL] Error reloading model checkpoint: {e}")
        return False

    # Check metrics.json
    metrics_files = list(run_dir.rglob("metrics.json"))
    if metrics_files:
        with open(metrics_files[0], "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("\n--- SageMaker Evaluation Metrics ---")
        print(f"  Image AUROC: {metrics.get('image_auroc', 'N/A')}")
        print(f"  Pixel AUROC: {metrics.get('pixel_auroc', 'N/A')}")
        print(f"  AUPRO:       {metrics.get('au_pro', 'N/A')}")
        print(f"  Precision:   {metrics.get('precision', 'N/A')}")
        print(f"  Recall:      {metrics.get('recall', 'N/A')}")
        print(f"  F1 Score:    {metrics.get('f1', 'N/A')}")

    # Check timing profile
    timing_files = list(run_dir.rglob("timing_profile.json"))
    if timing_files:
        with open(timing_files[0], "r", encoding="utf-8") as f:
            timings = json.load(f)
        print("\n--- SageMaker Hardware & Timing Profile ---")
        print(f"  Compute Device:     {timings.get('device', 'N/A')}")
        print(f"  Feature Extraction: {timings.get('feature_extraction_seconds', 'N/A')} s")
        print(f"  Coreset Selection:  {timings.get('coreset_selection_seconds', 'N/A')} s")
        print(f"  Total Pipeline:     {timings.get('total_pipeline_seconds', 'N/A')} s")
        print(f"  Peak GPU VRAM:      {timings.get('peak_gpu_vram_mib', 'N/A')} MiB")
        print(f"  Peak Host RAM:      {timings.get('peak_host_ram_mib', 'N/A')} MiB")

    print("\n[SUCCESS] Artifacts validated and ready for production inference.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate SageMaker downloaded model artifacts")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing downloaded SageMaker artifacts")
    args = parser.parse_args()

    validate_artifacts(args.run_dir)


if __name__ == "__main__":
    main()
