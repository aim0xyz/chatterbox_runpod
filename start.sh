#!/bin/bash
set -e

echo "[startup] Starting Qwen3-TTS worker..."

# --- FLASH ATTENTION INSTALLATION FROM VOLUME ---
FLASH_ATTN_WHEEL="/runpod-volume/qwen3_models/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

if [ -f "$FLASH_ATTN_WHEEL" ]; then
    echo "[startup] Installing Flash Attention..."
    python3 -m pip install "$FLASH_ATTN_WHEEL" --force-reinstall --no-deps
else
    echo "❌ CRITICAL: Flash Attention wheel missing at $FLASH_ATTN_WHEEL!"
    exit 1
fi

# Start the handler
echo "[startup] Starting Qwen3-TTS Handler..."
cd /app || exit 1
python3 -u /app/handler.py
