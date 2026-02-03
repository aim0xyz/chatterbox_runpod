#!/bin/bash

# --- FLASH ATTENTION SIDE-LOAD ---
# Detect any Flash Attention wheel in the volume.
WHL_FILE=$(ls /runpod-volume/qwen3_models/flash_attn*.whl 2>/dev/null | head -n 1)
if [ -n "$WHL_FILE" ]; then
    echo "[startup] Found Flash Attention wheel: $WHL_FILE"
    # TRICK: Pip requires at least 5 parts in a .whl filename (name-version-python-abi-platform)
    # We copy your file to a temporary valid name so pip won't complain.
    VALID_NAME="/tmp/flash_attn-2.7.4-cp310-cp310-linux_x86_64.whl"
    cp "$WHL_FILE" "$VALID_NAME"
    echo "[startup] Installing via temporary valid name: $VALID_NAME"
    python3 -m pip install "$VALID_NAME" --no-deps
else
    echo "[startup] No Flash Attention wheel found in /runpod-volume/qwen3_models/, skipping side-load."
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
