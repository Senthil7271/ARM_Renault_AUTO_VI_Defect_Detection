# Comprehensive AutoVI Dataset Audit Report

**Date**: 2026-11-14
**Evaluator**: Senior Industrial Vision Engineering Team
**Dataset**: Automotive Visual Inspection Dataset (AutoVI), version 1.0.0
**Zenodo DOI**: [10.5281/zenodo.10459003](https://doi.org/10.5281/zenodo.10459003)
**Workspace Root**: `C:\Users\linga\Downloads\ArmChallenge`

---

## 1. Executive Summary

A comprehensive automated audit of all 6 AutoVI product categories was conducted across thousands of industrial production images and defect annotations. All directories, image dimensions, color channels, ground truth masks, defect configs, cryptographic SHA-256 hashes, and cross-split integrity were verified.

### Category Breakdown & File Counts

| Product Category | Double-Nested Path | Train Good | Test Good | Test Defective | Test Total | GT Masks Total | Defect Classes Configured |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **pipe_clip** | `pipe_clip/pipe_clip` | 195 | 195 | 142 | 337 | 142 | 2 (`unclipped`, `operator`) |
| **engine_wiring** | `engine_wiring/engine_wiring` | 285 | 285 | 322 | 607 | 355 | 4 (`fastening`, `blue_hoop`, `cardboard`, `obstruction`) |
| **pipe_staple** | `pipe_staple/pipe_staple` | 191 | 188 | 117 | 305 | 117 | 1 (`missing`) |
| **tank_screw** | `tank_screw/tank_screw` | 318 | 318 | 95 | 413 | 95 | 1 (`missing`) |
| **underbody_pipes** | `underbody_pipes/underbody_pipes` | 161 | 161 | 184 | 345 | 185 | 2 (`obstruction`, `operator`) |
| **underbody_screw** | `underbody_screw/underbody_screw` | 373 | 374 | 18 | 392 | 18 | 1 (`missing`) |
| **TOTAL** | — | **1,523** | **1,521** | **878** | **2,399** | **912** | **11** |

---

## 2. Ground-Truth Mask Resolution & Multi-Mask Findings

1. **Resolution Pattern**: AutoVI does not place flat masks in `ground_truth/`. Rather, masks are nested under:
   `ground_truth/<defect_type>/<image_stem>/<mask_id>.png`
   where each test defective image `test/<defect_type>/<image_stem>.png` has a dedicated folder containing 1 or more mask files.
2. **Missing Masks**: **0 missing masks**. Every defective image has verified corresponding ground-truth mask files.
3. **Multi-Region & Multi-Defect Union**: In categories such as `engine_wiring` (33 `multiple` images with 66 masks) and `underbody_pipes` (1 `multiple` image with 2 masks), individual mask files isolate distinct defect regions. Our `MaskProcessor` uses bitwise OR union to reconstruct the complete binary ground truth for pixel evaluation.
4. **Pixel Values & Multiclass Conventions**:
   - `pipe_clip`: `unclipped` has pixel value 255; `operator` has pixel value 254.
   - `engine_wiring`: `fastening` (255), `blue_hoop` (254), `cardboard` (253), `obstruction` (252).
   - Masks are preserved without threshold corruption via nearest-neighbor interpolation.

---

## 3. Cryptographic Duplicate & Cross-Split Leakage Analysis

1. **Cryptographic Hashes**: SHA-256 hashes were computed for every image in the dataset.
2. **Cross-Split Data Leakage**:
   - **`pipe_clip`**: **0 leaked images** (100% clean train/test boundary).
   - **`pipe_staple`**, **`tank_screw`**, **`underbody_pipes`**, **`underbody_screw`**: **0 leaked images**.
   - **`engine_wiring`**: **60 test/good images** were found to have identical cryptographic SHA-256 hashes to images in `train/good` (e.g. `test/good/0092.png` == `train/good/0117.png`). This empirical finding has been fully logged in `outputs/dataset_audit/data_leakage_report.csv` without modifying the official files.
3. **Intra-Category Duplicates**: 73 duplicate hash groups were detected and cataloged in `outputs/dataset_audit/duplicate_report.csv`.

---

## 4. Image Dimensions and Integrity

- **Corruption**: Zero unreadable or corrupted images across all splits.
- **Image Channels**: 3-channel RGB photographs across all categories.
- **Image Resolutions**:
  - `pipe_clip`: 400 × 400 pixels
  - `engine_wiring`: 400 × 400 pixels
  - `pipe_staple`: 400 × 400 pixels
  - `tank_screw`: 1200 × 1200 pixels
  - `underbody_pipes`: 1200 × 1200 pixels
  - `underbody_screw`: 1200 × 1200 pixels
- All masks strictly match their source image dimensions.
