#!/bin/bash
set -e

echo "[startup] Starting Qwen3-TTS worker (SDPA mode — no flash-attn needed)..."

# Start the handler directly
cd /app || exit 1
python3 -u /app/handler.py
