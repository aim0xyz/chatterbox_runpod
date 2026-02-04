#!/bin/bash
set -e

echo "[startup] Starting Qwen3-TTS worker..."

# --- FLASH ATTENTION INSTALLATION FROM VOLUME ---
# Install the Python 3.11 + PyTorch 2.5 compatible wheel from volume
FLASH_ATTN_WHEEL="/runpod-volume/qwen3_models/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

if [ -f "$FLASH_ATTN_WHEEL" ]; then
    echo "[startup] Verifying Flash Attention wheel compatibility..."
    python3 -c "
import sys, pkg_resources
try:
    # Validate Python version (3.11.x)
    py_ver = sys.version_info
    assert py_ver.major == 3 and py_ver.minor == 11, f'Python version mismatch! Expected 3.11.x, found {py_ver.major}.{py_ver.minor}'
    
    # Validate PyTorch version (2.5.x)
    torch_ver = pkg_resources.parse_version('$(python3 -c \"import torch; print(torch.__version__)\")')
    assert torch_ver.major == 2 and torch_ver.minor == 5, f'PyTorch version mismatch! Expected 2.5.x, found {torch_ver}'
    
    print(f'✅ Wheel compatible: Python {py_ver.major}.{py_ver.minor}, PyTorch {torch_ver}')
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

# Start the RunPod handler
echo "[startup] Launching handler.py..."
python3 -u /app/handler.py
