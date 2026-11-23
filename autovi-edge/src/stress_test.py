"""AutoVI Edge: Model Stress-Test and Robustness Perturbation Engine.

Applies controlled physical perturbations simulating real-world manufacturing conditions:
1. Conveyor mechanical rotation / angular skew
2. Vibration translation / jitter
3. Optical lens defocus / motion blur
4. Ambient lighting fluctuations / shadows
5. CMOS sensor grain / high-gain electrical noise
"""

from typing import Dict, Tuple
import cv2
import numpy as np


class StressTestLab:
    """Simulates real-world manufacturing line disturbances on inspection images."""

    @staticmethod
    def apply_perturbations(
        image_bgr: np.ndarray,
        rotation_deg: float = 0.0,
        shift_x: int = 0,
        shift_y: int = 0,
        blur_ksize: int = 0,
        illumination_pct: float = 0.0,
        noise_sigma: float = 0.0,
    ) -> Tuple[np.ndarray, Dict]:
        """Applies sequence of perturbations to input BGR image."""
        img = image_bgr.copy()
        h, w = img.shape[:2]

        applied = []

        # 1. Rotation (Conveyor angular skew)
        if abs(rotation_deg) > 0.1:
            center = (w // 2, h // 2)
            M_rot = cv2.getRotationMatrix2D(center, float(rotation_deg), 1.0)
            img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)
            applied.append(f"Rotation: {rotation_deg:+.1f}°")

        # 2. Translation (Mechanical conveyor vibration / offset)
        if abs(shift_x) > 0 or abs(shift_y) > 0:
            M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)
            applied.append(f"Jitter: ({shift_x:+d}px, {shift_y:+d}px)")

        # 3. Gaussian Blur (Lens defocus / motion blur)
        if blur_ksize > 0:
            k = int(blur_ksize) if int(blur_ksize) % 2 == 1 else int(blur_ksize) + 1
            img = cv2.GaussianBlur(img, (k, k), 0)
            applied.append(f"Blur: {k}x{k} Gaussian")

        # 4. Illumination / Brightness (Factory lighting variance)
        if abs(illumination_pct) > 0.5:
            factor = 1.0 + (float(illumination_pct) / 100.0)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            applied.append(f"Lighting: {illumination_pct:+.0f}%")

        # 5. Sensor Noise (CMOS sensor grain / electrical interference)
        if noise_sigma > 0.5:
            noise = np.random.normal(0, float(noise_sigma), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            applied.append(f"Sensor Noise: σ={noise_sigma:.1f}")

        description = ", ".join(applied) if applied else "Nominal (No Perturbations Applied)"

        metadata = {
            "rotation_deg": float(rotation_deg),
            "shift_x": int(shift_x),
            "shift_y": int(shift_y),
            "blur_ksize": int(blur_ksize),
            "illumination_pct": float(illumination_pct),
            "noise_sigma": float(noise_sigma),
            "applied_count": len(applied),
            "description": description,
        }

        return img, metadata

    @staticmethod
    def compute_robustness_metrics(
        score_base: float,
        score_stressed: float,
        tau: float,
        is_anomaly_base: bool,
        is_anomaly_stressed: bool,
    ) -> Dict:
        """Computes score drift, stability index, and classification flipping."""
        drift = score_stressed - score_base
        drift_pct = (drift / max(score_base, 1e-4)) * 100.0

        # Classification flip detection
        flipped = is_anomaly_base != is_anomaly_stressed
        if flipped:
            if not is_anomaly_base and is_anomaly_stressed:
                flip_type = "FALSE_ALARM"  # Nominal part falsely flagged due to perturbation
            else:
                flip_type = "MISSED_DEFECT"  # Defect masked by perturbation
        else:
            flip_type = "STABLE"

        # Stability index (100% = perfectly invariant, decays with score deviation)
        # 10% drift -> 90% stability
        stability = max(0.0, min(100.0, 100.0 - abs(drift_pct)))

        return {
            "score_base": round(score_base, 2),
            "score_stressed": round(score_stressed, 2),
            "threshold": round(tau, 2),
            "score_drift": round(drift, 2),
            "score_drift_pct": round(drift_pct, 1),
            "stability_index": round(stability, 1),
            "is_flipped": flipped,
            "flip_type": flip_type,
            "margin_remaining": round(tau - score_stressed, 2),
        }
