#!/bin/bash

# Qwen3-TTS Startup Script for RunPod

echo "[startup] Starting Qwen3-TTS Handler..."

# Force working directory to /app where the code lives
cd /app || exit 1

# 1. Ensure dependencies are installed
# We use /app/requirements.txt specifically
pip install --no-cache-dir -r /app/requirements.txt

# 2. Check if model exists
if [ ! -f "/qwen3_models/model.safetensors" ]; then
    echo "[ERROR] Model files (specifically model.safetensors) not found in /qwen3_models!"
    echo "Please ensure you have uploaded the model files from S3 to that directory."
    # We don't exit here, in case they are using a different filename, but handler.py will likely fail later.
fi

# 3. Start the RunPod handler
echo "[startup] Launching handler.py..."
python3 -u /app/handler.py
