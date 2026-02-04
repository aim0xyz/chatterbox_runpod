#!/bin/bash
set -e

echo "[startup] Starting Qwen3-TTS worker..."

# --- FLASH ATTENTION INSTALLATION FROM VOLUME ---
# Install the Python 3.12 compatible wheel from volume
FLASH_ATTN_WHEEL="/runpod-volume/qwen3_models/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

if [ -f "$FLASH_ATTN_WHEEL" ]; then
    echo "[startup] Verifying Flash Attention wheel compatibility..."
    python3 -c "
import sys, pkg_resources
try:
    torch_ver = pkg_resources.parse_version('$(python3 -c \"import torch; print(torch.__version__)\")')
    assert torch_ver.major == 2 and torch_ver.minor == 9, f'PyTorch version mismatch! Expected 2.9.x, found {torch_ver}'
    print(f'✅ Wheel compatible with PyTorch {torch_ver}')
except Exception as e:
    print(f'❌ Validation error: {e}')
    sys.exit(1)
"
    python3 -m pip install "$FLASH_ATTN_WHEEL" --force-reinstall --no-deps
else
    echo "❌ CRITICAL: Flash Attention wheel missing at $FLASH_ATTN_WHEEL!"
    exit 1  # Fail fast instead of silent degradation
fi

# Start the handler
echo "[startup] Starting Qwen3-TTS Handler..."

# Force working directory to /app where the code lives
cd /app || exit 1

# Dependencies are pre-installed in the Docker image (including flash-attn built from source)

# Start the RunPod handler
echo "[startup] Launching handler.py..."
python3 -u /app/handler.py
