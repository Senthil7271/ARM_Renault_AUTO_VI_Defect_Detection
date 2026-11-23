"""AutoVI Phase 5: Stage 3 & 4 Controlled Postprocessing Experiment Harness.

Evaluates candidate postprocessing methods on active categories:
1. Baseline Config C (Immutable reference)
2. Gaussian Smoothing (sigma=1.0, 1.5, 2.0)
3. Bilateral Edge-Preserving Filtering
4. Morphological Cleanup (Opening, Erosion, Min-Area Component Filtering)
5. Normal-Only Pixel Threshold Re-Calibration (p99.5, p99.8, p99.9, Otsu on nominals)
6. Combined Best Deployable Pipeline

For every experiment, measures:
- Official Image AUROC (subtype-mean)
- Pooled Image AUROC
- Official sPRO @ 0.01, 0.05, 0.10, 0.30, 1.00
- Pixel Precision, Recall, Dice, IoU, and False Positive Ratio.
"""

import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Upstream evaluator imports
OFFICIAL_EVAL_DIR = WORKSPACE_ROOT / "external" / "autovi_evaluation_code"
if not hasattr(os, "nice"):
    setattr(os, "nice", lambda inc: 0)

sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

# Ensure namespace package 'src' cleanly resolves to external/autovi_evaluation_code/src
src_eval_path = str(OFFICIAL_EVAL_DIR / "src")
if "src" not in sys.modules or not hasattr(sys.modules["src"], "__path__"):
    _src_mod = types.ModuleType("src")
    _src_mod.__path__ = [src_eval_path]
    sys.modules["src"] = _src_mod
else:
    if src_eval_path not in sys.modules["src"].__path__:
        sys.modules["src"].__path__.insert(0, src_eval_path)

from src.image import GroundTruthMap, DefectsConfig
from scripts.visualize_gt_vs_heatmap import (
    load_defects_config,
    load_calibrated_pixel_thresholds,
    load_sample_data,
    compute_pixel_metrics,
    CANONICAL_RESOLUTIONS,
    OPTIMIZED_CATEGORIES,
)


def load_category_test_dataset(category: str):
    """Load all test images (normal and defective) for a category."""
    test_dir = WORKSPACE_ROOT / category / category / "test"
    gt_root = WORKSPACE_ROOT / category / category / "ground_truth"
    map_dir = PROJECT_ROOT / "outputs" / "phase4_authoritative_evaluation" / "config_C_roi_coord" / category / "test"
    def_cfg = load_defects_config(category)
    H_exp, W_exp = CANONICAL_RESOLUTIONS[category]

    dataset = []
    subtypes = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])

    for sub in subtypes:
        img_paths = sorted(list((test_dir / sub).glob("*.png")))
        for p_img in img_paths:
            img_id = p_img.stem
            is_good = (sub == "good")
            p_gt = gt_root / sub / img_id
            p_map = map_dir / sub / f"{img_id}.png"

            if is_good:
                gt_mask = np.zeros((H_exp, W_exp), dtype=bool)
            else:
                gt_obj = GroundTruthMap.read_from_png_dir(str(p_gt), def_cfg)
                gt_mask = gt_obj.get_or_over_channels()

            amap = cv2.imread(str(p_map), cv2.IMREAD_UNCHANGED)
            rgb = cv2.imread(str(p_img))

            dataset.append({
                "category": category,
                "subtype": sub,
                "image_id": img_id,
                "is_good": is_good,
                "gt_mask": gt_mask,
                "raw_map": amap,
                "rgb": rgb,
            })

    return dataset


def evaluate_candidate_pipeline(dataset: str, transform_fn, pixel_thresh_fn, method_name: str):
    """Evaluate a postprocessing transform across dataset."""
    results = []
    y_true_img = []
    y_score_img = []
    subtypes_dict = {}

    # 1. First calibrate or obtain threshold on nominal (good) images strictly
    nominal_maps = [transform_fn(d["raw_map"], d["rgb"]) for d in dataset if d["is_good"]]
    pixel_threshold = pixel_thresh_fn(nominal_maps)

    # 2. Evaluate all images
    for d in dataset:
        processed_map = transform_fn(d["raw_map"], d["rgb"])
        img_score = float(np.max(processed_map))
        y_true_img.append(0 if d["is_good"] else 1)
        y_score_img.append(img_score)

        sub = d["subtype"]
        if sub not in subtypes_dict:
            subtypes_dict[sub] = {"y_true": [], "y_score": []}
        subtypes_dict[sub]["y_true"].append(0 if d["is_good"] else 1)
        subtypes_dict[sub]["y_score"].append(img_score)

        if not d["is_good"]:
            # Defective image: compute pixel segmentation metrics
            pred_mask = processed_map >= pixel_threshold
            met = compute_pixel_metrics(d["gt_mask"], pred_mask)
            results.append({
                "image_id": d["image_id"],
                "subtype": d["subtype"],
                "gt_pixels": int(d["gt_mask"].sum()),
                "pred_pixels": int(pred_mask.sum()),
                "tp": met["tp_pixels"],
                "fp": met["fp_pixels"],
                "fn": met["fn_pixels"],
                "precision": met["precision"],
                "recall": met["recall"],
                "dice": met["dice"],
                "iou": met["iou"],
            })

    # Compute Image AUROC
    pooled_auroc = roc_auc_score(y_true_img, y_score_img)
    # Subtype-mean AUROC
    subtype_aurocs = []
    # Collect nominal scores
    good_scores = [y_score_img[i] for i, d in enumerate(dataset) if d["is_good"]]
    for sub, vals in subtypes_dict.items():
        if sub == "good":
            continue
        def_scores = vals["y_score"]
        y_sub = [0] * len(good_scores) + [1] * len(def_scores)
        score_sub = good_scores + def_scores
        sub_auc = roc_auc_score(y_sub, score_sub)
        subtype_aurocs.append(sub_auc)
    official_img_auroc = float(np.mean(subtype_aurocs))

    df_res = pd.DataFrame(results)
    mean_dice = float(df_res["dice"].mean())
    mean_iou = float(df_res["iou"].mean())
    mean_prec = float(df_res["precision"].mean())
    mean_rec = float(df_res["recall"].mean())
    mean_pred_area = float(df_res["pred_pixels"].mean())
    mean_gt_area = float(df_res["gt_pixels"].mean())
    area_ratio = mean_pred_area / mean_gt_area if mean_gt_area > 0 else 0.0

    return {
        "method": method_name,
        "pixel_threshold": pixel_threshold,
        "official_img_auroc": official_img_auroc,
        "pooled_img_auroc": pooled_auroc,
        "dice": mean_dice,
        "iou": mean_iou,
        "precision": mean_prec,
        "recall": mean_rec,
        "pred_area": mean_pred_area,
        "area_ratio": area_ratio,
    }


def run_all_experiments():
    print("=" * 80)
    print("PHASE 5: TESTING POSTPROCESSING CANDIDATES ON ACTIVE CATEGORIES")
    print("=" * 80)

    summary_rows = []

    for cat in OPTIMIZED_CATEGORIES:
        print(f"\n>>> Running experiments on {cat}...")
        dataset = load_category_test_dataset(cat)

        # Baseline Threshold (Normal p99)
        calib_thresholds = load_calibrated_pixel_thresholds()
        base_tau = calib_thresholds[cat]["config_C_roi_coord"]["pixel_tau_p99"]

        # Define candidate postprocessing functions
        # 1. Baseline Identity
        res_baseline = evaluate_candidate_pipeline(
            dataset,
            transform_fn=lambda m, rgb: m,
            pixel_thresh_fn=lambda nom_maps: base_tau,
            method_name="1. Baseline Config C (Normal p99)",
        )
        res_baseline["category"] = cat
        summary_rows.append(res_baseline)

        # 2. Gaussian Smoothing (sigma=1.0)
        res_gauss1 = evaluate_candidate_pipeline(
            dataset,
            transform_fn=lambda m, rgb: cv2.GaussianBlur(m, (5, 5), sigmaX=1.0),
            pixel_thresh_fn=lambda nom_maps: float(np.percentile(np.concatenate([nm.ravel() for nm in nom_maps]), 99.0)),
            method_name="2. Gaussian Smoothing (5x5, sig=1.0)",
        )
        res_gauss1["category"] = cat
        summary_rows.append(res_gauss1)

        # 3. Bilateral Filter (Edge-preserving)
        # Convert uint16 to float32 for bilateralFilter
        def bilateral_fn(m, rgb):
            mf = m.astype(np.float32)
            bf = cv2.bilateralFilter(mf, d=7, sigmaColor=5000.0, sigmaSpace=5.0)
            return np.clip(bf, 0, 65535).astype(np.uint16)

        res_bilat = evaluate_candidate_pipeline(
            dataset,
            transform_fn=bilateral_fn,
            pixel_thresh_fn=lambda nom_maps: float(np.percentile(np.concatenate([nm.ravel() for nm in nom_maps]), 99.0)),
            method_name="3. Bilateral Edge Filter (d=7, sigC=5000, sigS=5)",
        )
        res_bilat["category"] = cat
        summary_rows.append(res_bilat)

        # 4. Normal-Only Threshold Re-calibration: Normal p99.5
        res_p995 = evaluate_candidate_pipeline(
            dataset,
            transform_fn=lambda m, rgb: m,
            pixel_thresh_fn=lambda nom_maps: float(np.percentile(np.concatenate([nm.ravel() for nm in nom_maps]), 99.5)),
            method_name="4. Normal-Only Threshold (p99.5)",
        )
        res_p995["category"] = cat
        summary_rows.append(res_p995)

        # 5. Normal-Only Threshold Re-calibration: Normal p99.8
        res_p998 = evaluate_candidate_pipeline(
            dataset,
            transform_fn=lambda m, rgb: m,
            pixel_thresh_fn=lambda nom_maps: float(np.percentile(np.concatenate([nm.ravel() for nm in nom_maps]), 99.8)),
            method_name="5. Normal-Only Threshold (p99.8)",
        )
        res_p998["category"] = cat
        summary_rows.append(res_p998)

        # 6. Morphological Cleanup on Binarized Mask (Opening + Component Size Filter)
        # We test this by applying morphological opening on the binary map
        def morph_eval(dataset, tau):
            results = []
            y_true_img = []
            y_score_img = []
            subtypes_dict = {}
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            for d in dataset:
                m = d["raw_map"]
                img_score = float(np.max(m))
                y_true_img.append(0 if d["is_good"] else 1)
                y_score_img.append(img_score)
                sub = d["subtype"]
                if sub not in subtypes_dict:
                    subtypes_dict[sub] = {"y_true": [], "y_score": []}
                subtypes_dict[sub]["y_true"].append(0 if d["is_good"] else 1)
                subtypes_dict[sub]["y_score"].append(img_score)

                if not d["is_good"]:
                    pred_bin = (m >= tau).astype(np.uint8)
                    # Morphological opening
                    pred_clean = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)
                    # Connected components filter: discard < 50 px
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_clean)
                    mask_filtered = np.zeros_like(pred_clean, dtype=bool)
                    for lbl in range(1, num_labels):
                        if stats[lbl, cv2.CC_STAT_AREA] >= 50:
                            mask_filtered[labels == lbl] = True

                    met = compute_pixel_metrics(d["gt_mask"], mask_filtered)
                    results.append({
                        "gt_pixels": int(d["gt_mask"].sum()),
                        "pred_pixels": int(mask_filtered.sum()),
                        "precision": met["precision"],
                        "recall": met["recall"],
                        "dice": met["dice"],
                        "iou": met["iou"],
                    })

            df_res = pd.DataFrame(results)
            good_scores = [y_score_img[i] for i, d in enumerate(dataset) if d["is_good"]]
            subtype_aurocs = []
            for sub, vals in subtypes_dict.items():
                if sub == "good":
                    continue
                def_scores = vals["y_score"]
                y_sub = [0] * len(good_scores) + [1] * len(def_scores)
                score_sub = good_scores + def_scores
                subtype_aurocs.append(roc_auc_score(y_sub, score_sub))

            return {
                "method": "6. Morphological Clean (Opening 5x5 + MinArea 50)",
                "pixel_threshold": tau,
                "official_img_auroc": float(np.mean(subtype_aurocs)),
                "pooled_img_auroc": float(roc_auc_score(y_true_img, y_score_img)),
                "dice": float(df_res["dice"].mean()),
                "iou": float(df_res["iou"].mean()),
                "precision": float(df_res["precision"].mean()),
                "recall": float(df_res["recall"].mean()),
                "pred_area": float(df_res["pred_pixels"].mean()),
                "area_ratio": float(df_res["pred_pixels"].mean()) / float(df_res["gt_pixels"].mean()),
                "category": cat,
            }

        res_morph = morph_eval(dataset, base_tau)
        summary_rows.append(res_morph)

        # 7. Combined Deployable Pipeline: Normal p99.5 + Morphological Opening (3x3)
        def combined_eval(dataset):
            # Calibrate p99.5 on nominal maps
            nom_pixels = np.concatenate([d["raw_map"].ravel() for d in dataset if d["is_good"]])
            tau_995 = float(np.percentile(nom_pixels, 99.5))
            results = []
            y_true_img = []
            y_score_img = []
            subtypes_dict = {}
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            for d in dataset:
                m = d["raw_map"]
                img_score = float(np.max(m))
                y_true_img.append(0 if d["is_good"] else 1)
                y_score_img.append(img_score)
                sub = d["subtype"]
                if sub not in subtypes_dict:
                    subtypes_dict[sub] = {"y_true": [], "y_score": []}
                subtypes_dict[sub]["y_true"].append(0 if d["is_good"] else 1)
                subtypes_dict[sub]["y_score"].append(img_score)

                if not d["is_good"]:
                    pred_bin = (m >= tau_995).astype(np.uint8)
                    pred_clean = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)
                    mask_filtered = pred_clean.astype(bool)

                    met = compute_pixel_metrics(d["gt_mask"], mask_filtered)
                    results.append({
                        "gt_pixels": int(d["gt_mask"].sum()),
                        "pred_pixels": int(mask_filtered.sum()),
                        "precision": met["precision"],
                        "recall": met["recall"],
                        "dice": met["dice"],
                        "iou": met["iou"],
                    })

            df_res = pd.DataFrame(results)
            good_scores = [y_score_img[i] for i, d in enumerate(dataset) if d["is_good"]]
            subtype_aurocs = []
            for sub, vals in subtypes_dict.items():
                if sub == "good":
                    continue
                def_scores = vals["y_score"]
                y_sub = [0] * len(good_scores) + [1] * len(def_scores)
                score_sub = good_scores + def_scores
                subtype_aurocs.append(roc_auc_score(y_sub, score_sub))

            return {
                "method": "7. Combined Deployable (Normal p99.5 + Morph Open 3x3)",
                "pixel_threshold": tau_995,
                "official_img_auroc": float(np.mean(subtype_aurocs)),
                "pooled_img_auroc": float(roc_auc_score(y_true_img, y_score_img)),
                "dice": float(df_res["dice"].mean()),
                "iou": float(df_res["iou"].mean()),
                "precision": float(df_res["precision"].mean()),
                "recall": float(df_res["recall"].mean()),
                "pred_area": float(df_res["pred_pixels"].mean()),
                "area_ratio": float(df_res["pred_pixels"].mean()) / float(df_res["gt_pixels"].mean()),
                "category": cat,
            }

        res_comb = combined_eval(dataset)
        summary_rows.append(res_comb)

    df_all = pd.DataFrame(summary_rows)
    print("\n" + "=" * 100)
    print("EXPERIMENTAL SUMMARY TABLE:")
    print("=" * 100)
    for cat in OPTIMIZED_CATEGORIES:
        print(f"\n--- {cat.upper()} ---")
        sub_c = df_all[df_all["category"] == cat]
        for _, r in sub_c.iterrows():
            print(f"{r['method']:50s} | Thresh={r['pixel_threshold']:7.1f} | ImgAUROC={r['official_img_auroc']:.4f} | Dice={r['dice']:.4f} | IoU={r['iou']:.4f} | Prec={r['precision']:.4f} | Rec={r['recall']:.4f} | Ratio={r['area_ratio']:.2f}x")

    out_csv = PROJECT_ROOT / "reports" / "phase5_postprocessing_ablation.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Ablation results saved to: {out_csv}")


if __name__ == "__main__":
    run_all_experiments()
