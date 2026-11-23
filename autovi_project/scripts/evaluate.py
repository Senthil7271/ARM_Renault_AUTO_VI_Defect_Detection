"""Evaluation script for trained AutoVI PatchCore models.

Usage:
    python scripts/evaluate.py --category pipe_clip
    python scripts/evaluate.py --category all
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.loader import create_autovi_dataloaders
from src.evaluation.evaluator import ModelEvaluator
from src.models.patchcore_model import PatchCoreModel
from src.visualization.heatmaps import Visualizer

ALL_CATEGORIES = [
    "pipe_clip",
    "engine_wiring",
    "pipe_staple",
    "tank_screw",
    "underbody_pipes",
    "underbody_screw",
]


def evaluate_single_category(category: str, args, default_cfg: dict) -> dict:
    exp_dir = Path(args.experiments) / category
    model_path = exp_dir / "model" / "patchcore_model.pkl"

    if not model_path.exists():
        print(f"[SKIP] Model artifact not found at {model_path}. Train this category first.")
        return {"category": category, "status": "not_run"}

    print(f"\n==================================================")
    print(f"Evaluating Model for Category: {category}")
    print(f"==================================================")

    model = PatchCoreModel.load(model_path)
    target_size = tuple(default_cfg.get("data", {}).get("image_size", [256, 256]))

    _, _, test_loader, meta = create_autovi_dataloaders(
        workspace_root=args.workspace,
        category=category,
        target_size=target_size,
        batch_size=args.batch_size,
        num_workers=0,
    )

    evaluator = ModelEvaluator(model=model, output_dir=exp_dir)
    results = evaluator.evaluate(test_loader)

    print("\n--- Evaluation Results ---")
    print(f"Image AUROC: {results['image_auroc']:.4f}")
    print(f"Image AP:    {results['image_ap']:.4f}")
    print(f"Pixel AUROC: {results['pixel_auroc']:.4f}")
    print(f"Pixel AP:    {results['pixel_ap']:.4f}")
    print(f"AUPRO:       {results.get('au_pro', 0.0):.4f}")
    print(f"Threshold:   {results['threshold']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"F1:          {results['f1']:.4f}")
    print(f"FPR:         {results['fpr']:.4f}")
    print("Confusion Matrix:", results["confusion_matrix"])

    # Generate four-panel visualizations
    if args.visualize_count > 0:
        print(f"\nGenerating up to {args.visualize_count} four-panel visualization examples...")
        viz = Visualizer(output_dir=exp_dir / "visualizations")
        df_preds = pd.read_csv(exp_dir / "predictions.csv")

        # Pick diverse samples: true positive, false negative, false positive, true negative
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
        mask_proc = MaskProcessor(target_size=None)
        cat_root = meta["category_root"]

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

            # Get predicted heatmap
            from PIL import Image
            import torchvision.transforms as T
            import torch
            pil_img = Image.open(img_p).convert("RGB")
            tr = T.Compose([
                T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
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

        print(f"Visualizations saved to: {exp_dir / 'visualizations'}")

    results["category"] = category
    results["status"] = "completed"
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore models on AutoVI test set")
    parser.add_argument(
        "--category",
        type=str,
        default="pipe_clip",
        help="Category name (or 'all' for all 6 categories)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(PROJECT_ROOT.parent),
        help="Workspace root containing categories",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "experiments"),
        help="Directory where experiment models are stored",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size",
    )
    parser.add_argument(
        "--visualize-count",
        type=int,
        default=8,
        help="Number of 4-panel comparison examples to generate",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)

    if args.category.lower() == "all":
        comparison_rows = []
        for cat in ALL_CATEGORIES:
            res = evaluate_single_category(cat, args, default_cfg)
            comparison_rows.append(res)

        df_comp = pd.DataFrame(comparison_rows)
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        df_comp.to_csv(reports_dir / "six_category_comparison.csv", index=False)
        print(f"\n[DONE] Six-category comparison saved to: {reports_dir / 'six_category_comparison.csv'}")
    else:
        evaluate_single_category(args.category, args, default_cfg)


if __name__ == "__main__":
    main()
