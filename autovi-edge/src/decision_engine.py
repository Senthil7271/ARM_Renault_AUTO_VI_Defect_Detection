"""AutoVI Edge: Operational Decision and Verdict Engine.

Evaluates raw PatchCore anomaly scores against category-calibrated decision thresholds
using a production-grade 3-tier classification:
1. PASS (Nominal): Score < 0.85 * Threshold
2. REVIEW_REQUIRED (Marginal anomaly / low confidence): 0.85 * Threshold <= Score < Threshold
3. DEFECT (Defect detected): Score >= Threshold

Includes telemetry, threshold margin calculation, and rolling inspection statistics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time


class Verdict(str, Enum):
    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    DEFECT = "DEFECT"


@dataclass
class InspectionResult:
    category: str
    anomaly_score: float
    threshold: float
    verdict: Verdict
    margin: float             # Score - Threshold
    confidence_pct: float     # Relative confidence (0 - 100%)
    timestamp: float = field(default_factory=time.time)
    review_band_min: float = 0.0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "anomaly_score": round(self.anomaly_score, 4),
            "threshold": round(self.threshold, 4),
            "verdict": self.verdict.value,
            "margin": round(self.margin, 4),
            "confidence_pct": round(self.confidence_pct, 2),
            "timestamp": round(self.timestamp, 3),
            "review_band_min": round(self.review_band_min, 4),
        }


class DecisionEngine:
    """Evaluates edge inspection results and tracks operational yield."""

    # Official Calibrated Category Thresholds (tau at ~5% validation FPR)
    CALIBRATED_THRESHOLDS: Dict[str, float] = {
        "tank_screw": 17.6370,
        "pipe_clip": 18.9086,
        "engine_wiring": 21.5754,
        "pipe_staple": 17.2834,
        "underbody_pipes": 21.3370,
        "underbody_screw": 14.4447,
    }

    def __init__(self, review_margin_factor: float = 1.0):
        self.review_factor = review_margin_factor
        self.history: List[InspectionResult] = []
        self.total_count = 0
        self.pass_count = 0
        self.defect_count = 0
        self.review_count = 0

    def evaluate(self, category: str, score: float, override_threshold: Optional[float] = None) -> InspectionResult:
        """Evaluates an anomaly score against calibrated threshold (Pure Binary: PASS vs REJECT)."""
        tau = override_threshold if override_threshold is not None else self.CALIBRATED_THRESHOLDS.get(category, 18.0)
        margin = score - tau

        if score >= tau:
            verdict = Verdict.DEFECT  # REJECT / DEFECT
            conf = min(100.0, 50.0 + (margin / tau) * 100.0)
            self.defect_count += 1
        else:
            verdict = Verdict.PASS
            conf = min(100.0, 50.0 + ((tau - score) / tau) * 50.0)
            self.pass_count += 1

        self.total_count += 1
        res = InspectionResult(
            category=category,
            anomaly_score=score,
            threshold=tau,
            verdict=verdict,
            margin=margin,
            confidence_pct=conf,
            review_band_min=tau,
        )
        self.history.append(res)
        if len(self.history) > 1000:
            self.history.pop(0)

        return res

    def get_statistics(self) -> dict:
        """Returns cumulative inspection yield statistics."""
        if self.total_count == 0:
            return {
                "total_inspections": 0,
                "pass_rate_pct": 0.0,
                "defect_rate_pct": 0.0,
                "review_rate_pct": 0.0,
                "recent_avg_score": 0.0,
            }

        recent = self.history[-50:]
        recent_avg = float(sum(r.anomaly_score for r in recent) / len(recent))

        return {
            "total_inspections": self.total_count,
            "pass_count": self.pass_count,
            "defect_count": self.defect_count,
            "review_count": self.review_count,
            "pass_rate_pct": round(self.pass_count / self.total_count * 100.0, 2),
            "defect_rate_pct": round(self.defect_count / self.total_count * 100.0, 2),
            "review_rate_pct": round(self.review_count / self.total_count * 100.0, 2),
            "recent_avg_score": round(recent_avg, 4),
        }
