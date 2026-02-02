#!/bin/bash

# Qwen3-TTS Startup Script for RunPod

echo "[startup] Starting Qwen3-TTS Handler..."

# Force working directory to /app where the code lives
cd /app || exit 1

# 1. Ensure dependencies are installed
# We use /app/requirements.txt specifically
pip install --no-cache-dir -r /app/requirements.txt

# 2. Check if model exists, download if missing
if [ ! -f "/qwen3_models/model.safetensors" ]; then
    echo "[startup] Model files not found in /qwen3_models, downloading..."
    python3 /app/download_models.py
fi

# 3. Start the RunPod handler
echo "[startup] Launching handler.py..."
python3 -u /app/handler.py
