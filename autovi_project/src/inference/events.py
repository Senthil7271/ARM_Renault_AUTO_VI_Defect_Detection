"""Unknown Defect Event Detection and Logging Module.

Generates structured JSON records when abnormal products are detected.
Note: defect_class remains strictly 'unknown' in accordance with open-set detection principles.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid


class DefectEventLogger:
    """Logs structured unknown-defect anomaly events."""

    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.event_counter = 0

    def generate_event(
        self,
        category: str,
        anomaly_score: float,
        threshold: Optional[float],
        is_anomaly: bool,
        heatmap_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        bounding_boxes: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Creates a standardized anomaly event record."""
        self.event_counter += 1
        event_id = f"ANOMALY_{self.event_counter:06d}_{uuid.uuid4().hex[:6]}"

        timestamp = datetime.now(timezone.utc).isoformat()

        event = {
            "event_id": event_id,
            "product_category": category,
            "prediction": "anomaly" if is_anomaly else "normal",
            "anomaly_score": round(float(anomaly_score), 4),
            "score_threshold": round(float(threshold), 4) if threshold is not None else None,
            "defect_class": "unknown",  # Always unknown in open-set unsupervised detection
            "localization_available": (heatmap_path is not None),
            "bounding_boxes": bounding_boxes or [],
            "heatmap_path": heatmap_path,
            "mask_path": mask_path,
            "timestamp": timestamp,
        }

        # Save single event file
        event_file = self.log_dir / f"{event_id}.json"
        with open(event_file, "w", encoding="utf-8") as f:
            json.dump(event, f, indent=2)

        return event
