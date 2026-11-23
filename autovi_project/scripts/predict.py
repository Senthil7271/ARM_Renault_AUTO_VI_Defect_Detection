"""Single image and directory inference script.

Usage:
    python scripts/predict.py --category pipe_clip --image "path/to/image.png"
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.events import DefectEventLogger
from src.inference.predictor import AnomalyPredictor


def main():
    parser = argparse.ArgumentParser(description="Run PatchCore inference on inspection images")
    parser.add_argument(
        "--category",
        type=str,
        default="pipe_clip",
        help="Category name",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model .pkl file (defaults to outputs/experiments/<category>/model/patchcore_model.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "predictions"),
        help="Directory to save visual predictions and events",
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else (
        PROJECT_ROOT / "outputs" / "experiments" / args.category / "model" / "patchcore_model.pkl"
    )

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        sys.exit(1)

    event_logger = DefectEventLogger(log_dir=Path(args.output) / "events")
    predictor = AnomalyPredictor(
        model=model_path,
        event_logger=event_logger,
        category=args.category,
    )

    result = predictor.predict_image(
        image_path=args.image,
        output_dir=args.output,
        save_visuals=True,
    )

    print("\n================ Inference Result ================")
    print(f"Image:         {result['image_path']}")
    print(f"Category:      {args.category}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"Threshold:     {result['threshold']:.4f}")
    print(f"Prediction:    {result['prediction'].upper()}")
    print(f"Bounding Boxes: {len(result['bounding_boxes'])} localized defect region(s)")
    if result.get("heatmap_path"):
        print(f"Heatmap:       {result['heatmap_path']}")
        print(f"Overlay:       {result['overlay_path']}")
        print(f"Binary Mask:   {result['mask_path']}")
    if result.get("event"):
        print(f"Logged Event:  {result['event']['event_id']}")
    print("==================================================")


if __name__ == "__main__":
    main()
