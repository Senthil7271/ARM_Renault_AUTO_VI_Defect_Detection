"""AutoVI Edge: Web Dashboard and Inspection Server.

Provides a responsive real-time web dashboard for Raspberry Pi 5 + Hailo-8 AI HAT+:
- Live MJPEG video stream with blended anomaly heatmaps and HUD banner.
- Real-time 3-tier verdict display (PASS / REVIEW_REQUIRED / DEFECT).
- Dynamic category switching across all 6 AutoVI inspection tasks.
- Latency telemetry (NPU feature extraction, NN search, end-to-end FPS).
- REST API for industrial PLC / automation integration.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

# Add source directory to path
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from src.camera import CameraAcquisition
from src.decision_engine import DecisionEngine, Verdict
from src.heatmap import HeatmapGenerator
from src.inference_engine import EdgeAnomalyEngine

app = Flask(__name__)

# Global Components
camera = None
engine = None
decision_engine = None
heatmap_gen = None
active_category = "tank_screw"
stream_overlay = True


def get_html_dashboard() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoVI Edge - Raspberry Pi 5 + Hailo-8</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-green: #22c55e;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-blue: #38bdf8;
            --border: #334155;
        }
        body {
            margin: 0;
            padding: 0;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .header {
            padding: 16px 24px;
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .badge {
            font-size: 0.75rem;
            padding: 4px 8px;
            border-radius: 4px;
            background: #0284c7;
            color: white;
            font-weight: 600;
        }
        .main-container {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 20px;
            padding: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .card {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .video-box {
            position: relative;
            width: 100%;
            border-radius: 6px;
            overflow: hidden;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .video-box img {
            width: 100%;
            height: auto;
            display: block;
        }
        .verdict-banner {
            margin-top: 16px;
            padding: 16px;
            border-radius: 6px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 1px;
            background: #1e293b;
            border: 2px solid var(--border);
            transition: all 0.3s ease;
        }
        .verdict-PASS { background: rgba(34, 197, 94, 0.15); border-color: var(--accent-green); color: var(--accent-green); }
        .verdict-REVIEW_REQUIRED { background: rgba(245, 158, 11, 0.15); border-color: var(--accent-amber); color: var(--accent-amber); }
        .verdict-DEFECT { background: rgba(239, 68, 68, 0.15); border-color: var(--accent-red); color: var(--accent-red); }
        .control-group {
            margin-bottom: 16px;
        }
        .control-group label {
            display: block;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
            font-weight: 600;
        }
        select, button {
            width: 100%;
            padding: 10px 12px;
            background: #0f172a;
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            cursor: pointer;
        }
        select:focus, button:hover {
            border-color: var(--accent-blue);
        }
        .telemetry-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.9rem;
        }
        .telemetry-val {
            font-weight: 700;
            color: var(--accent-blue);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        .stat-item {
            background: #0f172a;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-num {
            font-size: 1.25rem;
            font-weight: 800;
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span>AutoVI Edge Vision</span>
            <span class="badge">Raspberry Pi 5 + Hailo-8</span>
        </h1>
        <div id="backend-label" style="font-size: 0.85rem; color: var(--accent-blue);">Initializing backend...</div>
    </div>

    <div class="main-container">
        <!-- Live Video & Visual Feed -->
        <div class="card">
            <div class="video-box">
                <img src="/video_feed" alt="Live Inspection Feed">
            </div>
            <div id="verdict-banner" class="verdict-banner verdict-PASS">READY</div>
        </div>

        <!-- Inspection Controls & Telemetry -->
        <div>
            <div class="card" style="margin-bottom: 20px;">
                <h3 style="margin-top: 0; font-size: 1.05rem;">Inspection Control</h3>
                <div class="control-group">
                    <label>Inspection Category</label>
                    <select id="cat-select" onchange="changeCategory(this.value)">
                        <option value="tank_screw">tank_screw (Active Config C)</option>
                        <option value="pipe_clip">pipe_clip (Active Config C)</option>
                        <option value="engine_wiring">engine_wiring (Active Config C)</option>
                        <option value="pipe_staple">pipe_staple (Frozen Baseline)</option>
                        <option value="underbody_pipes">underbody_pipes (Frozen Baseline)</option>
                        <option value="underbody_screw">underbody_screw (Frozen Baseline)</option>
                    </select>
                </div>
                <div class="control-group">
                    <button onclick="toggleOverlay()">Toggle Heatmap Overlay</button>
                </div>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; font-size: 1.05rem;">Edge Telemetry</h3>
                <div class="telemetry-row">
                    <span>Inspection Latency</span>
                    <span id="latency-val" class="telemetry-val">-- ms</span>
                </div>
                <div class="telemetry-row">
                    <span>Throughput (FPS)</span>
                    <span id="fps-val" class="telemetry-val">-- FPS</span>
                </div>
                <div class="telemetry-row">
                    <span>Current Anomaly Score</span>
                    <span id="score-val" class="telemetry-val">--</span>
                </div>
                <div class="telemetry-row">
                    <span>Threshold (&tau;)</span>
                    <span id="tau-val" class="telemetry-val">--</span>
                </div>

                <div class="stats-grid">
                    <div class="stat-item">
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Pass Rate</div>
                        <div id="pass-rate" class="stat-num" style="color: var(--accent-green);">--%</div>
                    </div>
                    <div class="stat-item">
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Defect Rate</div>
                        <div id="defect-rate" class="stat-num" style="color: var(--accent-red);">--%</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateTelemetry() {
            fetch('/api/telemetry')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('backend-label').innerText = d.backend || 'Active';
                    document.getElementById('latency-val').innerText = (d.latency_ms || 0).toFixed(1) + ' ms';
                    document.getElementById('fps-val').innerText = (d.fps || 0).toFixed(1) + ' FPS';
                    document.getElementById('score-val').innerText = (d.score || 0).toFixed(2);
                    document.getElementById('tau-val').innerText = (d.threshold || 0).toFixed(2);

                    const banner = document.getElementById('verdict-banner');
                    banner.innerText = d.verdict || 'MONITORING';
                    banner.className = 'verdict-banner verdict-' + (d.verdict || 'PASS');

                    if (d.stats) {
                        document.getElementById('pass-rate').innerText = d.stats.pass_rate_pct + '%';
                        document.getElementById('defect-rate').innerText = d.stats.defect_rate_pct + '%';
                    }
                })
                .catch(err => console.error(err));
        }

        function changeCategory(cat) {
            fetch('/api/category', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({category: cat})
            });
        }

        function toggleOverlay() {
            fetch('/api/toggle_overlay', {method: 'POST'});
        }

        setInterval(updateTelemetry, 500);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return get_html_dashboard()


latest_telemetry = {
    "verdict": "PASS",
    "score": 0.0,
    "threshold": 18.0,
    "latency_ms": 0.0,
    "fps": 0.0,
    "backend": "Initializing",
    "category": active_category,
    "stats": {},
}


def generate_frames():
    global latest_telemetry, stream_overlay
    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Run Edge Anomaly Inspection
        res = engine.inspect(frame)
        score = res["anomaly_score"]
        raw_2d = res["patch_distances_2d"]
        tau = res["threshold"]
        cat_range = res["global_range"]

        # Decision verdict
        verdict_res = decision_engine.evaluate(engine.current_category, score, tau)
        verdict_str = verdict_res.verdict.value

        # Heatmap Generation
        _, uint8_map = heatmap_gen.process(
            raw_distances=raw_2d,
            category=engine.current_category,
            target_shape=(frame.shape[0], frame.shape[1]),
            category_min_max=cat_range,
        )

        if stream_overlay:
            display_frame = heatmap_gen.render_overlay(
                frame_bgr=frame,
                uint8_map=uint8_map,
                category=engine.current_category,
                verdict=verdict_str,
                score=score,
                threshold=tau,
            )
        else:
            display_frame = frame

        # Update telemetry
        latest_telemetry = {
            "verdict": verdict_str,
            "score": score,
            "threshold": tau,
            "latency_ms": res["timings_ms"]["total"],
            "fps": res["timings_ms"]["fps"],
            "backend": res["backend"],
            "category": engine.current_category,
            "stats": decision_engine.get_statistics(),
        }

        # Encode MJPEG
        ret, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry")
def api_telemetry():
    return jsonify(latest_telemetry)


@app.route("/api/category", methods=["GET", "POST"])
def api_category():
    global active_category
    if request.method == "POST":
        data = request.get_json() or {}
        new_cat = data.get("category")
        if new_cat and new_cat in engine.category_models:
            engine.set_category(new_cat)
            active_category = new_cat
            return jsonify({"status": "success", "category": active_category})
        return jsonify({"status": "error", "message": "Invalid category"}), 400
    return jsonify({"category": active_category, "available": list(engine.category_models.keys())})


@app.route("/api/toggle_overlay", methods=["POST"])
def api_toggle_overlay():
    global stream_overlay
    stream_overlay = not stream_overlay
    return jsonify({"status": "success", "overlay_enabled": stream_overlay})


@app.route("/api/inspect", methods=["POST"])
def api_single_inspect():
    """Industrial trigger endpoint for single-frame inspection via REST."""
    ret, frame = camera.read()
    if not ret:
        return jsonify({"status": "error", "message": "Frame capture failed"}), 500
    res = engine.inspect(frame)
    verdict_res = decision_engine.evaluate(engine.current_category, res["anomaly_score"], res["threshold"])
    return jsonify({
        "status": "success",
        "result": verdict_res.to_dict(),
        "timings_ms": res["timings_ms"],
    })


def main():
    global camera, engine, decision_engine, heatmap_gen, active_category
    parser = argparse.ArgumentParser(description="AutoVI Edge Inspection Server")
    parser.add_argument("--category", type=str, default="tank_screw", help="Initial category")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "hailo", "onnx"], help="Inference backend")
    parser.add_argument("--camera", type=str, default="0", help="Camera index, picam2, or dataset path")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host")
    args = parser.parse_args()

    print("=" * 80)
    print("STARTING AUTOVI EDGE SERVER (Raspberry Pi 5 + Hailo-8)")
    print("=" * 80)

    # 1. Initialize Engine
    model_dir = APP_DIR / "models"
    engine = EdgeAnomalyEngine(model_dir=model_dir, preferred_backend=args.backend)

    # 2. Load ROI Configs
    roi_cfg_file = APP_DIR / "configs" / "roi_configurations.json"
    roi_data = {}
    if roi_cfg_file.exists():
        with open(roi_cfg_file, "r", encoding="utf-8") as f:
            roi_data = json.load(f).get("categories", {})
    heatmap_gen = HeatmapGenerator(roi_configs=roi_data)

    # 3. Decision Engine
    decision_engine = DecisionEngine(review_margin_factor=0.85)

    # 4. Camera Acquisition
    cam_src = int(args.camera) if args.camera.isdigit() else args.camera
    camera = CameraAcquisition(source=cam_src, resolution=(640, 480), framerate=30)

    # Set Initial Category
    active_category = args.category
    engine.set_category(active_category)
    print(f"\n[SERVER READY] Inspection service running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
