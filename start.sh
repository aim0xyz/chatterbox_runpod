#!/bin/bash
set -e

echo "============================================"
echo "[startup] Chatterbox TTS Handler v1.3"
echo "[startup] Patching Chatterbox library..."
echo "============================================"

CHATTERBOX_PATH="/app/chatterbox"

if [ ! -d "$CHATTERBOX_PATH" ]; then
    CHATTERBOX_PATH=$(python3 -c "import chatterbox; import os; print(os.path.dirname(chatterbox.__file__))" 2>/dev/null || echo "")
fi

if [ -z "$CHATTERBOX_PATH" ] || [ ! -d "$CHATTERBOX_PATH" ]; then
    echo "[startup] WARNING: Chatterbox path not found, skipping patch"
    exec python3 -u /handler.py
fi

echo "[startup] Chatterbox path: $CHATTERBOX_PATH"

ANALYZER_FILE="$CHATTERBOX_PATH/models/alignment_stream_analyzer.py"
if [ -f "$ANALYZER_FILE" ]; then
    echo "[startup] Patching: $ANALYZER_FILE"
    cp "$ANALYZER_FILE" "${ANALYZER_FILE}.backup"
    sed -i 's/>=\s*2/>=50/g' "$ANALYZER_FILE"
    sed -i 's/==\s*2/==50/g' "$ANALYZER_FILE"
    sed -i 's/>\s*2/>50/g' "$ANALYZER_FILE"
    sed -i 's/< 3/< 51/g' "$ANALYZER_FILE"
    sed -i 's/token_repetition=True/token_repetition=False/g' "$ANALYZER_FILE"
    echo "[startup] Patched alignment_stream_analyzer.py"
else
    echo "[startup] alignment_stream_analyzer.py not found"
fi

T3_FILE="$CHATTERBOX_PATH/models/t3.py"
if [ -f "$T3_FILE" ]; then
    echo "[startup] Patching: $T3_FILE"
    cp "$T3_FILE" "${T3_FILE}.backup"
    sed -i 's/token_repetition=True/token_repetition=False/g' "$T3_FILE"
    sed -i 's/>=\s*2/>=50/g' "$T3_FILE"
    echo "[startup] Patched t3.py"
else
    echo "[startup] t3.py not found"
fi

echo "============================================"
echo "[startup] Starting handler..."
echo "============================================"

exec python3 -u /handler.py