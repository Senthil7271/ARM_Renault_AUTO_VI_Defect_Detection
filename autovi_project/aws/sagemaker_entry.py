"""SageMaker Training Entry Point for AutoVI PatchCore.

Executes inside SageMaker PyTorch Deep Learning Container on GPU (e.g. ml.g4dn.xlarge).
Handles:
- Standard SageMaker environment variables (SM_CHANNEL_TRAIN, SM_MODEL_DIR, SM_OUTPUT_DATA_DIR)
- GPU/CUDA acceleration with automatic fallback
- High-resolution timing and memory profiling
- Normal-only PatchCore training & threshold calibration
- Complete test set evaluation and visualization artifact generation
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Defense-in-depth: Ensure NumPy 1.x compatibility with PyTorch 2.1.0 DLC (Container only)
is_sagemaker = any(k in os.environ for k in ("SM_TRAINING_ENV", "SM_MODEL_DIR", "SM_CHANNEL_TRAIN")) or Path("/opt/ml").exists()
if is_sagemaker:
    try:
        import numpy as _np
        _np_major = int(_np.__version__.split(".")[0])
        if _np_major >= 2:
            print(f"[RECOVERY] Incompatible NumPy {_np.__version__} detected in container.")
            print(f"[RECOVERY] Downgrading to numpy==1.26.4 and opencv-python-headless==4.9.0.80...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "numpy==1.26.4", "opencv-python-headless==4.9.0.80", "scipy<1.14.0"
            ])
            print(f"[RECOVERY] Environment repaired. Relaunching entrypoint...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as _err:
        pass

# Add project root and source tree to sys.path
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch
import yaml

from src.dataset.loader import create_autovi_dataloaders
from src.evaluation.evaluator import ModelEvaluator
from src.models.patchcore_model import PatchCoreModel
from src.visualization.heatmaps import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="AutoVI SageMaker Training Entry Point")

    # SageMaker standard path arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Root directory containing AutoVI categories",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
        help="Directory to save evaluation reports and plots",
    )

    # Model and training hyperparameters
    parser.add_argument("--category", type=str, default="pipe_clip", help="AutoVI category name")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="Feature extractor")
    parser.add_argument("--coreset-ratio", type=float, default=0.05, help="Coreset sampling ratio")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Distance calculation chunk size")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Candidate pool size")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--image-size", type=int, default=256, help="Input image dimension")
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Target false alarm rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--visualize-count", type=int, default=8, help="Number of 4-panel visual plots")

    return parser.parse_args()


def print_environment_banner(args, device: torch.device):
    print("\n" + "=" * 65)
    print("      AutoVI PatchCore - AWS SageMaker GPU Training      ")
    print("=" * 65)
    print(f"  PyTorch Version:  {torch.__version__}")
    print(f"  NumPy Version:    {np.__version__}")
    print(f"  CUDA Available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device:      {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM:       {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")
    print(f"  Execution Target: {device}")
    print(f"  Data Channel:     {args.data_dir}")
    print(f"  Model Output Dir: {args.model_dir}")
    print(f"  Reports Output:   {args.output_dir}")
    print(f"  Category:         {args.category}")
    print(f"  Coreset Ratio:    {args.coreset_ratio}")
    print(f"  Chunk Size:       {args.chunk_size}")
    print("=" * 65 + "\n")


def train_and_evaluate_category(category: str, args, device: torch.device):
    print(f"\n>>> Starting Processing for Category: {category} <<<")
    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir) / category
    output_dir = Path(args.output_dir) / category
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    timing_summary = {
        "category": category,
        "device": str(device),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t_global_start = time.perf_counter()

    # Step 1: Dataloaders
    t_load0 = time.perf_counter()
    train_loader, val_loader, test_loader, meta = create_autovi_dataloaders(
        workspace_root=data_dir,
        category=category,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        train_val_ratio=0.9,
        seed=args.seed,
        num_workers=args.num_workers,
        manifest_dir=output_dir / "manifests",
    )
    timing_summary["data_prep_seconds"] = round(time.perf_counter() - t_load0, 3)
    print(f"Data ready: {meta['num_train']} train, {meta['num_val']} val, {meta['num_test']} test samples.")

    # Step 2: Initialize Model
    model = PatchCoreModel(
        backbone_name=args.backbone,
        layers=("layer2", "layer3"),
        target_size=(args.image_size, args.image_size),
        coreset_ratio=args.coreset_ratio,
        num_neighbors=9,
        dimension_reduction=128,
        chunk_size=args.chunk_size,
        candidate_pool_size=args.candidate_pool_size,
        device=device,
    )

    # Step 3: Train / Fit Memory Bank
    print(f"\nBuilding PatchCore memory bank on {device}...")
    model.fit_memory_bank(train_loader, verbose=True)

    # Step 4: Calibrate Threshold
    t_cal0 = time.perf_counter()
    calib_threshold = model.calibrate_threshold(val_loader, target_fpr=args.target_fpr)
    calib_time = time.perf_counter() - t_cal0
    timing_summary["calibration_seconds"] = round(calib_time, 3)
    print(f"Calibrated decision threshold on validation set: {calib_threshold:.4f} (took {calib_time:.2f}s)")

    # Step 5: Save Model Artifact
    model_save_path = model_dir / "patchcore_model.pkl"
    print(f"Saving model artifact to: {model_save_path}")
    model.save(model_save_path)

    # Also save in root of SM_MODEL_DIR for standard SageMaker model packaging
    root_model_path = Path(args.model_dir) / "patchcore_model.pkl"
    model.save(root_model_path)

    # Step 6: Test Set Evaluation
    print(f"\nEvaluating on official test set ({meta['num_test']} samples)...")
    t_eval0 = time.perf_counter()
    evaluator = ModelEvaluator(model=model, output_dir=output_dir)
    results = evaluator.evaluate(test_loader)
    eval_time = time.perf_counter() - t_eval0
    timing_summary["evaluation_seconds"] = round(eval_time, 3)

    # Combine timing records
    timing_summary.update(model.timing_profile)
    timing_summary["total_pipeline_seconds"] = round(time.perf_counter() - t_global_start, 3)

    # Save timing profile
    with open(output_dir / "timing_profile.json", "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)

    # Copy metrics to model directory as well so it's packaged in model.tar.gz
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "-" * 50)
    print(f"  Results for {category}:")
    print(f"    Image AUROC: {results['image_auroc']:.4f}")
    print(f"    Image AP:    {results['image_ap']:.4f}")
    print(f"    Pixel AUROC: {results['pixel_auroc']:.4f}")
    print(f"    AUPRO:       {results.get('au_pro', 0.0):.4f}")
    print(f"    Precision:   {results['precision']:.4f}")
    print(f"    Recall:      {results['recall']:.4f}")
    print(f"    F1 Score:    {results['f1']:.4f}")
    print(f"    Total Time:  {timing_summary['total_pipeline_seconds']}s")
    print("-" * 50)

    # Step 7: Four-Panel Visualizations
    if args.visualize_count > 0:
        print(f"\nGenerating {args.visualize_count} four-panel visualization comparisons...")
        viz = Visualizer(output_dir=output_dir / "visualizations")
        import pandas as pd
        df_preds = pd.read_csv(output_dir / "predictions.csv")

        samples_to_plot = []
        for cond, group in [
            ("tp", df_preds[(df_preds["label_true"] == 1) & (df_preds["prediction"] == "anomaly")]),
            ("fn", df_preds[(df_preds["label_true"] == 1) & (df_preds["prediction"] == "normal")]),
            ("fp", df_preds[(df_preds["label_true"] == 0) & (df_preds["prediction"] == "anomaly")]),
            ("tn", df_preds[(df_preds["label_true"] == 0) & (df_preds["prediction"] == "normal")]),
        ]:
            if not group.empty:
                samples_to_plot.extend(group.head(max(1, args.visualize_count // 4)).to_dict("records"))

        from src.dataset.masks import MaskProcessor
        from PIL import Image
        import torchvision.transforms as T
        mask_proc = MaskProcessor(target_size=None)
        cat_root = meta["category_root"]

        tr = T.Compose([
            T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for row in samples_to_plot[:args.visualize_count]:
            img_p = Path(row["image_path"])
            st = row["defect_type"]
            lbl = row["label_true"]
            score = row["anomaly_score"]

            if lbl == 1:
                m_paths = mask_proc.find_mask_paths(cat_root, st, img_p.stem)
                gt_mask, _ = mask_proc.load_combined_mask(m_paths)
            else:
                gt_mask, _ = mask_proc.load_combined_mask([])

            pil_img = Image.open(img_p).convert("RGB")
            t_img = tr(pil_img).unsqueeze(0)
            _, hms = model.predict_batch(t_img)

            viz.create_four_panel(
                orig_img_path=img_p,
                gt_mask=gt_mask,
                pred_heatmap=hms[0],
                threshold=model.threshold,
                score=score,
                label_true=lbl,
                defect_type=st,
            )
        print("Visualizations saved successfully.")

    return results, timing_summary


def main():
    args = parse_args()

    # Determine device
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print_environment_banner(args, device)

    categories = [
        "pipe_clip",
        "pipe_staple",
        "engine_wiring",
        "tank_screw",
        "underbody_pipes",
        "underbody_screw",
    ]

    if args.category.lower() == "all":
        all_results = {}
        for cat in categories:
            res, timings = train_and_evaluate_category(cat, args, device)
            all_results[cat] = {"metrics": res, "timings": timings}

        # Save comparative summary
        summary_path = Path(args.output_dir) / "all_categories_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[DONE] All 6 categories completed. Summary saved to: {summary_path}")
    else:
        train_and_evaluate_category(args.category, args, device)
        print(f"\n[DONE] Training and evaluation complete for {args.category}.")


if __name__ == "__main__":
    main()
