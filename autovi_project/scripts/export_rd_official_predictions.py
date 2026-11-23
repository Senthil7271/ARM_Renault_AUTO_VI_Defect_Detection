"""Exports canonical uint16 PNG anomaly maps for Reverse Distillation recovered model."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.reverse_distillation import ReverseDistillationModel

def export_maps():
    model_path = PROJECT_ROOT / "outputs" / "phase3" / "recovered_pilot2" / "pipe_clip" / "reverse_distillation" / "calibrated_model.pth"
    model = ReverseDistillationModel.load(model_path, device="cpu")
    model.eval()

    manifest_p = PROJECT_ROOT / "outputs" / "official_autovi" / "predictions" / "pipe_clip" / "manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    out_map_dir = PROJECT_ROOT / "outputs" / "phase3" / "recovered_pilot2" / "pipe_clip" / "reverse_distillation" / "anomaly_maps" / "pipe_clip" / "test"

    transform_norm = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_maps = []
    print(f"Generating anomaly maps for {len(samples)} samples...")
    for s in samples:
        img = Image.open(s["image_path"]).convert("RGB")
        t_norm = transform_norm(img).unsqueeze(0)
        with torch.no_grad():
            _, amap = model.predict_anomaly(t_norm)
        amap_np = amap.squeeze(0).numpy()
        amap_canon = cv2.resize(amap_np, (400, 400), interpolation=cv2.INTER_LINEAR)
        raw_maps.append((s, amap_canon))

    min_val = min(m.min() for _, m in raw_maps)
    max_val = max(m.max() for _, m in raw_maps)
    print(f"Global anomaly map min: {min_val:.4f}, max: {max_val:.4f}")

    for s, m in raw_maps:
        if max_val > min_val:
            m_norm = (m - min_val) / (max_val - min_val)
        else:
            m_norm = np.zeros_like(m)
        m_uint16 = (m_norm * 65535.0).astype(np.uint16)
        
        dest = out_map_dir / s["defect_name"] / f"{s['image_stem']}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), m_uint16)

    print(f"Successfully exported {len(raw_maps)} canonical maps to {out_map_dir}")

if __name__ == "__main__":
    export_maps()
