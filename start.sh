#!/bin/bash
set -e

echo "[startup] Starting Qwen3-TTS worker..."

# --- FLASH ATTENTION INSTALLATION FROM VOLUME ---
FLASH_ATTN_WHEEL="/runpod-volume/qwen3_models/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

if [ -f "$FLASH_ATTN_WHEEL" ]; then
    echo "[startup] Verifying Flash Attention wheel compatibility..."
    # We import torch directly inside the check to avoid shell escaping issues
    python3 -c "
import sys, torch, pkg_resources
try:
    # 1. Validate Python version (3.11.x)
    py_ver = sys.version_info
    if not (py_ver.major == 3 and py_ver.minor == 11):
        print(f'❌ Python version mismatch! Expected 3.11.x, found {py_ver.major}.{py_ver.minor}')
        sys.exit(1)
    
    # 2. Validate PyTorch version (2.5.x)
    # Get the base version (e.g., 2.5.1 from 2.5.1+cu124)
    torch_str = torch.__version__.split('+')[0]
    v = pkg_resources.parse_version(torch_str)
    
    if v.major == 2 and v.minor == 5:
        print(f'✅ Compatibility Verified: Python {py_ver.major}.{py_ver.minor}, PyTorch {torch.__version__}')
    else:
        print(f'❌ PyTorch version mismatch! Expected 2.5.x, found {torch.__version__}')
        sys.exit(1)
except Exception as e:
    print(f'❌ Validation logic error: {e}')
    sys.exit(1)
"
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
