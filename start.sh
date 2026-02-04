#!/bin/bash
set -e

echo "[startup] Starting Qwen3-TTS worker..."

# --- FLASH ATTENTION INSTALLATION FROM VOLUME ---
# Install the Python 3.12 compatible wheel from volume
FLASH_ATTN_WHEEL="/runpod-volume/qwen3_models/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

if [ -f "$FLASH_ATTN_WHEEL" ]; then
    echo "[startup] Installing Flash Attention from volume: $FLASH_ATTN_WHEEL"
    python3 -m pip install "$FLASH_ATTN_WHEEL" --no-deps
else
    echo "[startup] ⚠️  Flash Attention wheel not found at: $FLASH_ATTN_WHEEL"
    echo "[startup] ⚠️  Model will run without Flash Attention (slower)"
fi

# Start the handler
echo "[startup] Starting Qwen3-TTS Handler..."

# Force working directory to /app where the code lives
cd /app || exit 1

# Dependencies are pre-installed in the Docker image (including flash-attn built from source)

# Start the RunPod handler
echo "[startup] Launching handler.py..."
python3 -u /app/handler.py
