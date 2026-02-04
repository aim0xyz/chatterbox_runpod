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

# Dependencies are now pre-installed in the Docker image for faster cold starts
# pip install --no-cache-dir -r /app/requirements.txt

# Start the RunPod handler
echo "[startup] Launching handler.py..."
python3 -u /app/handler.py
