"""Model Evaluation Module.

Evaluates trained PatchCore model on the official AutoVI test split.
Generates:
- metrics.json
- predictions.csv
- roc_curve.png
- pr_curve.png
- confusion_matrix.png
- Per-defect-type recall breakdown
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .metrics import compute_au_pro, compute_image_metrics, compute_pixel_metrics


class ModelEvaluator:
    """Evaluates PatchCore model predictions against ground truth on official test split."""

    def __init__(self, model: Any, output_dir: Path):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, test_loader: Any) -> Dict[str, Any]:
        """Runs test split evaluation and produces complete metric records and plots."""
        all_scores = []
        all_labels = []
        all_paths = []
        all_subtypes = []
        all_gt_masks = []
        all_pred_maps = []

        for batch in test_loader:
            images = batch["image"]
            masks = batch["mask"].numpy()  # (B, H, W)
            labels = batch["label"].numpy()  # (B,)
            paths = batch["image_path"]
            subtypes = batch["defect_type"]

            scores, maps = self.model.predict_batch(images)

            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())
            all_paths.extend(paths)
            all_subtypes.extend(subtypes)
            all_gt_masks.append(masks)
            all_pred_maps.append(maps)

        y_true = np.array(all_labels)
        y_score = np.array(all_scores)
        masks_true = np.concatenate(all_gt_masks, axis=0)  # (N, H, W)
        maps_pred = np.concatenate(all_pred_maps, axis=0)  # (N, H, W)

        # 1. Compute Image-Level Metrics
        threshold = self.model.threshold
        img_metrics = compute_image_metrics(y_true, y_score, threshold)

        # 2. Compute Pixel-Level Metrics
        pix_metrics = compute_pixel_metrics(masks_true, maps_pred, threshold=threshold)

        # 3. Compute AUPRO
        au_pro = compute_au_pro(masks_true, maps_pred, max_fpr=0.3)
        pix_metrics["au_pro"] = au_pro

        # 4. Compute Per-Subtype Detection Recall
        y_pred = (y_score >= threshold).astype(int)
        df_preds = pd.DataFrame({
            "image_path": all_paths,
            "defect_type": all_subtypes,
            "label_true": y_true,
            "anomaly_score": y_score,
            "prediction": np.where(y_pred == 1, "anomaly", "normal"),
            "correct": (y_pred == y_true),
        })

        subtype_recalls = {}
        for st in sorted(df_preds["defect_type"].unique()):
            sub_df = df_preds[df_preds["defect_type"] == st]
            if st == "good":
                # Specificity (True negative rate)
                subtype_recalls["good_specificity"] = float((sub_df["prediction"] == "normal").mean())
            else:
                subtype_recalls[f"recall_{st}"] = float((sub_df["prediction"] == "anomaly").mean())

        # Save predictions CSV
        df_preds.to_csv(self.output_dir / "predictions.csv", index=False)

        # Combine all metrics
        results = {
            **img_metrics,
            **pix_metrics,
            "subtype_performance": subtype_recalls,
            "num_test_samples": len(y_true),
            "num_test_good": int((y_true == 0).sum()),
            "num_test_defective": int((y_true == 1).sum()),
        }

        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Generate plots
        self._plot_roc_curve(y_true, y_score, img_metrics["image_auroc"])
        self._plot_pr_curve(y_true, y_score, img_metrics["image_ap"])
        self._plot_confusion_matrix(y_true, y_pred)

        return results

    def _plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray, auroc: float):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auroc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Image-Level ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve.png", dpi=150)
        plt.close()

    def _plot_pr_curve(self, y_true: np.ndarray, y_score: np.ndarray, ap: float):
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, color="blue", lw=2, label=f"PR curve (AP = {ap:.4f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Image-Level Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.output_dir / "pr_curve.png", dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Normal", "Anomaly"])
        plt.yticks(tick_marks, ["Normal", "Anomaly"])

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                plt.text(
                    j, i, format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close()
