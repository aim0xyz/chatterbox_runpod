#!/bin/bash
set -e

echo "============================================"
echo "[startup] Patching Chatterbox library..."
echo "============================================"

# Find the chatterbox installation path
CHATTERBOX_PATH=$(python3 -c "import chatterbox; import os; print(os.path.dirname(chatterbox.__file__))" 2>/dev/null || echo "")

if [ -z "$CHATTERBOX_PATH" ]; then
    echo "[startup] ERROR: Chatterbox not found!"
    # Try alternate path for editable install
    CHATTERBOX_PATH="/app/chatterbox"
    if [ ! -d "$CHATTERBOX_PATH" ]; then
        echo "[startup] Continuing without patching..."
        exec python3 -u /handler.py
    fi
fi

echo "[startup] Chatterbox path: $CHATTERBOX_PATH"

# Patch alignment_stream_analyzer.py
ANALYZER_FILE="$CHATTERBOX_PATH/models/alignment_stream_analyzer.py"
if [ -f "$ANALYZER_FILE" ]; then
    echo "[startup] Patching: $ANALYZER_FILE"
    
    # Backup
    cp "$ANALYZER_FILE" "${ANALYZER_FILE}.original"
    
    # Replace repetition threshold of 2 with 50
    sed -i 's/>=\s*2/>=50/g' "$ANALYZER_FILE"
    sed -i 's/==\s*2/==50/g' "$ANALYZER_FILE"
    sed -i 's/>\s*2/>50/g' "$ANALYZER_FILE"
    sed -i 's/< 3/< 51/g' "$ANALYZER_FILE"
    
    # Disable token_repetition forcing
    sed -i 's/token_repetition=True/token_repetition=False/g' "$ANALYZER_FILE"
    
    # Show what was changed
    echo "[startup] Changes made to alignment_stream_analyzer.py:"
    diff "${ANALYZER_FILE}.original" "$ANALYZER_FILE" || true
    
    echo "[startup] ✅ Patched alignment_stream_analyzer.py"
else
    echo "[startup] ⚠️  alignment_stream_analyzer.py not found at $ANALYZER_FILE"
    # List directory to help debug
    echo "[startup] Contents of $CHATTERBOX_PATH/models/:"
    ls -la "$CHATTERBOX_PATH/models/" 2>/dev/null || echo "Directory not found"
fi

# Patch t3.py
T3_FILE="$CHATTERBOX_PATH/models/t3.py"
if [ -f "$T3_FILE" ]; then
    echo "[startup] Patching: $T3_FILE"
    
    cp "$T3_FILE" "${T3_FILE}.original"
    
    # Disable token_repetition
    sed -i 's/token_repetition=True/token_repetition=False/g' "$T3_FILE"
    sed -i 's/>=\s*2/>=50/g' "$T3_FILE"
    
    echo "[startup] ✅ Patched t3.py"
else
    echo "[startup] ⚠️  t3.py not found at $T3_FILE"
fi

echo "============================================"
echo "[startup] Starting handler..."
echo "============================================"

exec python3 -u /handler.py