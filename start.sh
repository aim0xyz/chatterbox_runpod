#!/bin/bash

# --- FLASH ATTENTION SIDE-LOAD ---
# Try v2.8.3 first (latest), fallback to v2.7.4
WHL_FILE="/runpod-volume/qwen3_models/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl"
if [ ! -f "$WHL_FILE" ]; then
    WHL_FILE="/runpod-volume/qwen3_models/flash_attn-2.7.4-cp310-cp310-linux_x86_64.whl"
fi

if [ -f "$WHL_FILE" ]; then
    echo "[startup] Found Flash Attention wheel: $WHL_FILE"
    # Create temp valid name and install
    TEMP_WHL="/tmp/$(basename $WHL_FILE)"
    cp "$WHL_FILE" "$TEMP_WHL"
    echo "[startup] Installing via temporary valid name: $TEMP_WHL"
    python3 -m pip install "$TEMP_WHL" --no-deps
else
    echo "[startup] No Flash Attention wheel found, will try to install from PyPI"
fi

# Start the handler
echo "[startup] Starting Qwen3-TTS Handler..."

# Force working directory to /app where the code lives
cd /app || exit 1

# 1. Dependencies are now pre-installed in the Docker image for faster cold starts
# pip install --no-cache-dir -r /app/requirements.txt

# 2. Comprehensive Model Search
echo "[debug] --- System Path Search ---"
echo "[debug] Checking root /qwen3_models:"
ls -F /qwen3_models 2>/dev/null || echo "Folder /qwen3_models does not exist"

echo "[debug] Checking /runpod-volume:"
ls -F /runpod-volume 2>/dev/null || echo "Folder /runpod-volume does not exist"

echo "[debug] Searching for model.safetensors across all drives..."
find / -name "model.safetensors" 2>/dev/null | xargs dirname | head -n 1 > /tmp/model_path.txt
FOUND_PATH=$(cat /tmp/model_path.txt)

if [ -n "$FOUND_PATH" ]; then
    echo "[debug] FOUND MODEL AT: $FOUND_PATH"
else
    echo "[ERROR] model.safetensors NOT FOUND ANYWHERE ON THE SERVER!"
fi

# 3. Start the RunPod handler
echo "[startup] Launching handler.py..."
python3 -u /app/handler.py
