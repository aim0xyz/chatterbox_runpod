"""
RunPod Serverless Handler for Chatterbox TTS
Works around aggressive repetition detection with small chunks.
"""

import os
import uuid
import base64
import io
import re
import random
import subprocess
import sys
import traceback
from pathlib import Path

import runpod
import torch
import soundfile as sf
import numpy as np

print(f"[startup] Python: {sys.version}")

# --- Import Chatterbox ---
Chatterbox = None
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
    print("[startup] ✅ Multilingual TTS imported")
except Exception as e:
    print(f"[warn] Multilingual failed: {e}")
    try:
        from chatterbox.tts import ChatterboxTTS as Chatterbox
        print("[startup] ✅ English TTS imported")
    except Exception as e2:
        print(f"[ERROR] All imports failed: {e2}")

# --- Configuration ---
MODEL_PATH = None
for p in [Path("/runpod-volume/models"), Path("/workspace/models"), Path("/models")]:
    if p.exists() and (list(p.glob("*.safetensors")) or list(p.glob("*.pt"))):
        MODEL_PATH = p
        break
MODEL_PATH = MODEL_PATH or Path("/runpod-volume/models")

VOICE_ROOT = Path("/runpod-volume/user_voices")
PRESET_ROOT = Path("/runpod-volume/preset_voices")
SAMPLE_RATE = 24000

VOICE_ROOT.mkdir(parents=True, exist_ok=True)
PRESET_ROOT.mkdir(parents=True, exist_ok=True)

print(f"[config] Models: {MODEL_PATH}")
print(f"[config] User voices: {VOICE_ROOT}")
print(f"[config] Presets: {PRESET_ROOT}")

# --- Model ---
model = None

def load_model():
    global model
    if model is None:
        if Chatterbox is None:
            raise RuntimeError("Chatterbox not available")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[model] Loading on {device}...")
        
        os.environ['CHATTERBOX_MODEL_PATH'] = str(MODEL_PATH)
        model = Chatterbox.from_pretrained(device)
        
        print("[model] ✅ Loaded successfully")
    return model

# --- Utilities ---

def get_user_dir(user_id: str) -> Path:
    safe = "".join(c for c in user_id if c.isalnum() or c in "_-")
    d = VOICE_ROOT / safe
    d.mkdir(parents=True, exist_ok=True)
    return d

def find_voice(user_id=None, filename=None, preset=None):
    if preset:
        p = PRESET_ROOT / preset
        if p.exists():
            return p
        for f in PRESET_ROOT.rglob(preset):
            return f
        # Try with common extensions
        for ext in ['.wav', '.mp3', '.flac']:
            p = PRESET_ROOT / f"{preset}{ext}"
            if p.exists():
                return p
    
    if user_id and filename:
        p = get_user_dir(user_id) / filename
        if p.exists():
            return p
    
    return None

def clean_text(text: str) -> str:
    replacements = {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u2014": " - ", "\u2013": "-",
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r'\s+', ' ', text).strip()

def split_into_sentences(text: str) -> list:
    """Split text into individual sentences."""
    if not text:
        return []
    
    # Add period if missing
    text = text.strip()
    if text and text[-1] not in '.!?':
        text += '.'
    
    # Split on sentence endings
    parts = re.split(r'(?<=[.!?])\s+', text)
    
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    
    return sentences

def chunk_sentences(sentences: list, max_chars: int = 180) -> list:
    """Combine sentences into chunks, respecting max length."""
    if not sentences:
        return []
    
    chunks = []
    current = ""
    
    for sent in sentences:
        # If single sentence is too long, split on commas
        if len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            
            # Split long sentence
            parts = sent.split(', ')
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if not current:
                    current = part
                elif len(current) + len(part) + 2 <= max_chars:
                    current = current + ", " + part
                else:
                    chunks.append(current)
                    current = part
        else:
            # Normal sentence
            if not current:
                current = sent
            elif len(current) + len(sent) + 1 <= max_chars:
                current = current + " " + sent
            else:
                chunks.append(current)
                current = sent
    
    if current:
        chunks.append(current)
    
    return [c.strip() for c in chunks if c.strip()]

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.squeeze(arr).flatten().astype(np.float32)

def apply_fade(wav, fade_ms=20):
    if len(wav) < 100:
        return wav
    
    wav = wav.copy()
    fade_samples = min(int(SAMPLE_RATE * fade_ms / 1000), len(wav) // 4)
    
    if fade_samples > 0:
        wav[:fade_samples] *= np.linspace(0, 1, fade_samples).astype(np.float32)
        wav[-fade_samples:] *= np.linspace(1, 0, fade_samples).astype(np.float32)
    
    return wav

def stitch_chunks(audio_list, pause_ms=300):
    if not audio_list:
        return np.array([], dtype=np.float32)
    
    if len(audio_list) == 1:
        return apply_fade(audio_list[0])
    
    pause = np.zeros(int(SAMPLE_RATE * pause_ms / 1000), dtype=np.float32)
    
    result = apply_fade(audio_list[0])
    for chunk in audio_list[1:]:
        result = np.concatenate([result, pause, apply_fade(chunk)])
    
    return result

# --- Handlers ---

def generate_tts_handler(job):
    try:
        inp = job.get("input", {})
        
        text = inp.get("text")
        if not text:
            return {"error": "text is required"}
        
        preset_voice = inp.get("preset_voice")
        user_id = inp.get("user_id")
        embedding_filename = inp.get("embedding_filename")
        language = inp.get("language", "en")
        
        # Generation parameters
        exaggeration = float(inp.get("exaggeration", 0.5))
        temperature = float(inp.get("temperature", 0.7))
        cfg_weight = float(inp.get("cfg_weight", 0.5))
        max_chunk_chars = int(inp.get("max_chunk_chars", 180))
        pause_ms = int(inp.get("pause_ms", 300))
        
        print(f"[tts] Text length: {len(text)} chars")
        print(f"[tts] Settings: temp={temperature}, exag={exaggeration}, cfg={cfg_weight}")
        
        # Get language from preset path if available
        if preset_voice and "/" in preset_voice:
            parts = preset_voice.split("/")
            if len(parts[0]) == 2:
                language = parts[0]
                print(f"[tts] Language from preset: {language}")
        
        # Find voice file
        voice_path = find_voice(user_id, embedding_filename, preset_voice)
        if not voice_path:
            return {"error": "Voice file not found"}
        
        print(f"[tts] Voice: {voice_path}")
        
        # Process text
        text = clean_text(text)
        sentences = split_into_sentences(text)
        chunks = chunk_sentences(sentences, max_chars=max_chunk_chars)
        
        print(f"[tts] Created {len(chunks)} chunks:")
        for i, c in enumerate(chunks):
            preview = c[:50] + "..." if len(c) > 50 else c
            print(f"[tts]   {i+1}. ({len(c)} chars) \"{preview}\"")
        
        if not chunks:
            return {"error": "No text to process"}
        
        # Load model
        tts = load_model()
        
        # Generate audio for each chunk
        audio_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            print(f"[tts] Generating chunk {i+1}/{len(chunks)}...")
            
            try:
                # Set seed for reproducibility but vary per chunk
                seed = (12345 + i * 9973) & 0xFFFFFFFF
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                # Generate
                output = tts.generate(
                    text=chunk_text,
                    language_id=language,
                    audio_prompt_path=str(voice_path),
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                    temperature=temperature,
                )
                
                wav = to_numpy(output)
                duration_ms = len(wav) / SAMPLE_RATE * 1000
                
                print(f"[tts]   ✓ {len(wav)} samples ({duration_ms:.0f}ms)")
                
                if len(wav) > 0:
                    audio_chunks.append(wav)
                else:
                    print(f"[tts]   ⚠ Empty output")
                    
            except Exception as e:
                print(f"[tts]   ✗ Error: {e}")
                traceback.print_exc()
        
        if not audio_chunks:
            return {"error": "No audio was generated"}
        
        print(f"[tts] Generated {len(audio_chunks)}/{len(chunks)} chunks")
        
        # Combine chunks
        final_wav = stitch_chunks(audio_chunks, pause_ms=pause_ms)
        
        # Normalize volume
        max_amp = np.max(np.abs(final_wav))
        if max_amp > 0.9:
            final_wav = final_wav * (0.9 / max_amp)
        
        # Encode to base64
        buf = io.BytesIO()
        sf.write(buf, final_wav, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        duration_sec = len(final_wav) / SAMPLE_RATE
        print(f"[tts] ✅ Complete: {duration_sec:.1f}s total")
        
        return {
            "status": "success",
            "audio": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "format": "wav",
            "duration_seconds": round(duration_sec, 2),
            "chunks_generated": len(audio_chunks),
            "chunks_requested": len(chunks),
        }
        
    except Exception as e:
        print(f"[error] TTS failed: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def clone_voice_handler(job):
    try:
        inp = job.get("input", {})
        audio_b64 = inp.get("audio_data")
        voice_name = inp.get("voice_name", "voice")
        user_id = inp.get("user_id", "anonymous")
        
        if not audio_b64:
            return {"error": "audio_data is required"}
        
        print(f"[clone] User: {user_id}, Voice: {voice_name}")
        
        user_dir = get_user_dir(user_id)
        
        # Save temp file
        temp_path = user_dir / f"temp_{uuid.uuid4().hex}.bin"
        with open(temp_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        
        # Convert with ffmpeg
        final_name = f"{voice_name}_{uuid.uuid4().hex[:6]}.wav"
        final_path = user_dir / final_name
        
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(temp_path),
                "-ar", "24000",
                "-ac", "1",
                str(final_path)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            temp_path.unlink(missing_ok=True)
            return {"error": f"FFmpeg failed: {e.stderr.decode() if e.stderr else str(e)}"}
        
        temp_path.unlink(missing_ok=True)
        
        print(f"[clone] ✅ Saved: {final_name}")
        
        return {
            "status": "success",
            "user_id": user_id,
            "embedding_filename": final_name,
            "voice_name": voice_name,
        }
        
    except Exception as e:
        print(f"[error] Clone failed: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def list_preset_voices_handler(job):
    try:
        target_lang = job.get("input", {}).get("language")
        
        print(f"[presets] Listing (filter: {target_lang or 'all'})")
        
        voices = []
        for path in PRESET_ROOT.rglob("*"):
            if path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                rel = str(path.relative_to(PRESET_ROOT))
                parts = rel.split("/")
                
                lang = "unknown"
                if len(parts) > 1 and len(parts[0]) == 2:
                    lang = parts[0]
                
                if target_lang and lang != "unknown" and lang != target_lang:
                    continue
                
                voices.append({
                    "filename": rel,
                    "name": path.stem.replace("_", " ").title(),
                    "language": lang,
                    "size_bytes": path.stat().st_size,
                })
        
        voices.sort(key=lambda x: (x['language'], x['name']))
        
        print(f"[presets] Found {len(voices)} voices")
        return {"status": "success", "preset_voices": voices, "count": len(voices)}
        
    except Exception as e:
        print(f"[error] List presets failed: {e}")
        return {"error": str(e)}

def list_user_voices_handler(job):
    try:
        user_id = job.get("input", {}).get("user_id")
        
        if not user_id:
            return {"error": "user_id is required"}
        
        print(f"[user_voices] User: {user_id}")
        
        user_dir = get_user_dir(user_id)
        
        if not user_dir.exists():
            return {"status": "success", "user_voices": [], "count": 0}
        
        voices = []
        for f in user_dir.iterdir():
            if f.suffix.lower() in ['.wav', '.mp3']:
                name = f.stem
                if "_" in name:
                    parts = name.split("_")
                    if len(parts) > 1:
                        name = " ".join(parts[:-1])
                
                voices.append({
                    "filename": f.name,
                    "name": name.title(),
                    "created_at": f.stat().st_mtime,
                    "size_bytes": f.stat().st_size,
                })
        
        voices.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        
        print(f"[user_voices] Found {len(voices)} voices")
        return {"status": "success", "user_voices": voices, "count": len(voices)}
        
    except Exception as e:
        print(f"[error] List user voices failed: {e}")
        return {"error": str(e)}

# --- Main Router ---

def handler(job):
    try:
        action = job.get("input", {}).get("action", "generate_tts")
        print(f"[handler] Action: {action}")
        
        if action == "clone_voice":
            return clone_voice_handler(job)
        elif action == "generate_tts":
            return generate_tts_handler(job)
        elif action == "list_presets":
            return list_preset_voices_handler(job)
        elif action == "list_user_voices":
            return list_user_voices_handler(job)
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        print(f"[fatal] {e}")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("[startup] RunPod Handler ready")
    runpod.serverless.start({"handler": handler})