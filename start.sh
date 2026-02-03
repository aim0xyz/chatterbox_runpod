#!/bin/bash

# --- FLASH ATTENTION SIDE-LOAD ---
# If you upload the .whl to your volume, we install it here instantly.
# RECOMMENDATION: Rename your uploaded file to 'flash_attn.whl' for this to work!
WHL_PATH="/runpod-volume/qwen3_models/flash_attn.whl"
if [ -f "$WHL_PATH" ]; then
    echo "[startup] Found Flash Attention wheel (flash_attn.whl) in volume. Installing..."
    python3 -m pip install "$WHL_PATH" --no-deps
else
    echo "[startup] No 'flash_attn.whl' found in volume, skipping side-load."
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
