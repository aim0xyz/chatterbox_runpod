#!/bin/bash

# Qwen3-TTS Startup Script for RunPod

echo "[startup] Starting Qwen3-TTS Handler..."

# Force working directory to /app where the code lives
cd /app || exit 1

# 1. Ensure dependencies are installed
# We use /app/requirements.txt specifically
pip install --no-cache-dir -r /app/requirements.txt

# 2. Check if model exists & Debug Paths
echo "[debug] Contents of /qwen3_models:"
ls -F /qwen3_models
if [ -d "/qwen3_models/qwen3_models" ]; then
    echo "[debug] Contents of /qwen3_models/qwen3_models:"
    ls -F /qwen3_models/qwen3_models
fi

if [ ! -f "/qwen3_models/model.safetensors" ] && [ ! -f "/qwen3_models/qwen3_models/model.safetensors" ]; then
    echo "[ERROR] Model files (model.safetensors) not found!"
    echo "Please check the [debug] logs above to see where the files went."
fi

# 3. Start the RunPod handler
echo "[startup] Launching handler.py..."
python3 -u /app/handler.py
