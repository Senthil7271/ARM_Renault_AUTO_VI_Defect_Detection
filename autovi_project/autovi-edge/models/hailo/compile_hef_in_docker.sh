#!/bin/bash
# Hailo DFC Compilation in Docker container
set -e
echo 'Starting Hailo DFC compilation inside Docker...'
hailo parser onnx /workspace/models/onnx/wideresnet50_patchcore.onnx --hw-arch hailo8 --output-har-path /workspace/models/hailo/parsed.har
hailo optimize /workspace/models/hailo/parsed.har --calib-set-path /workspace/models/hailo/calibration_dataset.npy --model-script /workspace/configs/wideresnet50_patchcore.alls --output-har-path /workspace/models/hailo/quantized.har
hailo compiler /workspace/models/hailo/quantized.har --hw-arch hailo8 --output-hef-path /workspace/models/hailo/wideresnet50_patchcore.hef
echo 'HEF compilation complete!'
