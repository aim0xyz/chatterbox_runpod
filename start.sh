#!/bin/bash

# Qwen3-TTS Startup Script for RunPod

echo "[startup] Starting Qwen3-TTS Handler..."

# 1. Ensure dependencies are installed
# Note: You might want to do this in your Dockerfile instead for faster boot
pip install -r requirements.txt

# 2. Check if model exists, download if missing
if [ ! -f "/qwen3_models/model.safetensors" ]; then
    echo "[startup] Model files not found in /qwen3_models, downloading..."
    python download_models.py
fi

# 3. Start the RunPod handler
echo "[startup] Launching handler.py..."
python -u handler.py
