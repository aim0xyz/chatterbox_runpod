"""
RunPod Serverless Handler for Chatterbox TTS - Enterprise Edition
Includes:
1. "Weird Noise" Fixes (Smart Trimming & Crossfade)
2. Mobile Audio Support (FFmpeg conversion)
3. Full Preset & User Voice Management
4. Verbose Logging & Stability Guards
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
import warnings
from pathlib import Path
import runpod

import torch
import soundfile as sf
import numpy as np

# --- 1. Environment & Import Setup ---
print(f"[startup] Python version: {sys.version}")

# Attempt to import Chatterbox with fallback logic
Chatterbox = None
try:
    print("[startup] Attempting to import ChatterboxMultilingualTTS...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
    print("[startup] ✅ Chatterbox Multilingual TTS imported successfully")
except Exception as e:
    print(f"[ERROR] Multilingual import failed: {e}")
    try:
        print("[startup] Attempting to import ChatterboxTTS (English only)...")
        from chatterbox.tts import ChatterboxTTS as Chatterbox
        print("[startup] ✅ Chatterbox TTS (English) imported successfully")
    except Exception as e2:
        print(f"[ERROR] English TTS import also failed: {e2}")
        Chatterbox = None

if Chatterbox is None:
    print("[ERROR] ⚠️  CHATTERBOX FAILED TO IMPORT - Handler will fail on requests")

# --- 2. Configuration & Paths ---

# diverse paths where models might be mounted in RunPod
possible_model_paths = [
    Path("/runpod-volume/models"),
    Path(os.getenv("MODEL_PATH", "/models")),
    Path("/workspace/models"),
    Path("/workspace/chatterbox-storage/models"),
    Path("/workspace"),
]

MODEL_PATH = None
for path in possible_model_paths:
    if path.exists() and path.is_dir():
        # Look for safetensors or pt files
        model_files = list(path.glob("*.safetensors")) + list(path.glob("*.pt"))
        if model_files:
            MODEL_PATH = path
            print(f"[config] ✅ Found models at: {MODEL_PATH}")
            break

if MODEL_PATH is None:
    print(f"[error] Model directory not found. Defaulting to /runpod-volume/models")
    MODEL_PATH = Path("/runpod-volume/models")

VOICE_EMBEDDING_ROOT = Path("/runpod-volume/user_voices")
PRESET_VOICES_ROOT = Path("/runpod-volume/preset_voices")
DEFAULT_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

# Ensure directories exist
VOICE_EMBEDDING_ROOT.mkdir(parents=True, exist_ok=True)
PRESET_VOICES_ROOT.mkdir(parents=True, exist_ok=True)

print(f"[config] Voice embedding root: {VOICE_EMBEDDING_ROOT}")
print(f"[config] Preset voices root: {PRESET_VOICES_ROOT}")

# --- 3. Global Model Loading (Lazy) ---
model = None

def load_model():
    """Lazy-load the Chatterbox model from disk."""
    global model
    if model is None:
        if Chatterbox is None:
            raise RuntimeError("Chatterbox library is missing/failed to import.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[startup] Using device: {device}")
        print(f"[startup] Loading model from: {MODEL_PATH}")

        try:
            os.environ['CHATTERBOX_MODEL_PATH'] = str(MODEL_PATH)
            model = Chatterbox.from_pretrained(device)
            print("[startup] Model loaded successfully.")
        except Exception as e:
            print(f"[startup] Model loading failed: {e}")
            raise RuntimeError(f"Failed to load Chatterbox model: {e}")
    return model

# --- 4. Helper Functions: Utilities ---

def derive_seed(base_seed: int, chunk_idx: int) -> int:
    """Deterministically derive a seed for each chunk to ensure consistency."""
    if not base_seed:
        return 0
    # Simple deterministic mixing
    return (base_seed + (chunk_idx * 10007)) & 0xFFFFFFFF

def get_user_voice_dir(user_id: str) -> Path:
    """Return a sanitized, user-specific directory path."""
    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ("_", "-"))
    user_dir = VOICE_EMBEDDING_ROOT / safe_user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def get_voice_audio_path(user_id: str = None, embedding_filename: str = None, preset_name: str = None) -> Path:
    """
    Smart resolver: Checks Presets first, then User ID folders.
    """
    # 1. Check Presets
    if preset_name:
        # Check exact match (e.g. "en/Male_1.wav")
        preset_path = PRESET_VOICES_ROOT / preset_name
        if preset_path.exists():
            print(f"[voice] Using preset voice: {preset_name}")
            return preset_path
        
        # Check flat match (e.g. "Male_1" -> "en/Male_1.wav")
        # This is a simple scan if the user didn't provide the language folder
        for f in PRESET_VOICES_ROOT.rglob(preset_name):
            print(f"[voice] Found fuzzy preset match: {f}")
            return f
            
    # 2. Check User Voice
    if user_id and embedding_filename:
        user_dir = get_user_voice_dir(user_id)
        audio_path = user_dir / embedding_filename
        if audio_path.exists():
            print(f"[voice] Using user voice: {user_id}/{embedding_filename}")
            return audio_path
        else:
            print(f"[warn] User voice file not found: {audio_path}")

    return None

# --- 5. Helper Functions: Text Processing ---

def clean_text_for_tts(text: str) -> str:
    """
    Normalize quotes and remove invisible characters.
    """
    replacements = {
        "\u201c": '"', "\u201d": '"', 
        "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-", 
        "\u2026": ".", "«": '"', "»": '"',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Remove weird whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences_safely(text: str):
    """
    Split text into sentences but keep quotes intact. 
    Prevents splitting "Hello," said the man.
    """
    if not re.search(r'[.!?]\s*$', text):
        text = text.strip() + "."

    # Regex Lookbehind for punctuation, but avoiding splitting inside quotes is complex.
    # This is a simplified robust splitter.
    parts = re.split(
        r'(?<=[.!?])\s+(?=(?:[^"“”«»]*["“”«»][^"“”«»]*["“”«»])*[^"“”«»]*$)',
        text
    )

    merged, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p: continue
        
        # Merge tiny fragments (e.g. "No.") into the next sentence to maintain flow
        if len(buf) + len(p) < 80:
            buf = (buf + " " + p).strip()
        else:
            if buf: merged.append(buf)
            buf = p
    if buf: merged.append(buf)
    return merged

def pack_sentences_into_chunks(sentences, max_chars=400):
    """Pack sentences into manageable chunks for the model."""
    chunks = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = f"{cur} {s}"
        else:
            chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())
    return chunks

def insert_quote_pauses(chunks, pause_tag="[pause] "):
    """
    Storyteller feature: If a chunk ends in a quote, add a pause marker.
    """
    out = chunks[:]
    for i in range(1, len(out)):
        if re.search(r'[\"“”»]$', out[i-1]):
            out[i] = pause_tag + out[i]
    return out

# --- 6. Helper Functions: Audio Processing (THE FIXES) ---

def to_numpy_1d(x) -> np.ndarray:
    """Ensure data is a flat float32 numpy array."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr.astype(np.float32)

def gentle_hp(wav_np, sample_rate=DEFAULT_SAMPLE_RATE, cutoff=40.0):
    """
    High-pass filter to remove sub-bass rumble (40Hz).
    """
    try:
        from scipy import signal
        wav = wav_np.astype(np.float32)
        sos = signal.butter(2, cutoff, 'hp', fs=sample_rate, output='sos')
        return signal.sosfilt(sos, wav).astype(np.float32)
    except ImportError:
        # If scipy is missing in the container
        return wav_np.astype(np.float32)

def trim_silence_and_artifacts(wav_np, threshold_db=-35, sample_rate=DEFAULT_SAMPLE_RATE):
    """
    THE FIX: Aggressively trims 'hallucinated' static/breathing from the ends of chunks.
    """
    if len(wav_np) == 0:
        return wav_np
        
    threshold_amp = 10 ** (threshold_db / 20)
    abs_wav = np.abs(wav_np)
    mask = abs_wav > threshold_amp
    
    # If audio is empty/silent
    if not np.any(mask):
        return np.zeros(int(0.1 * sample_rate), dtype=np.float32)

    # Find start (first sound)
    start_idx = np.argmax(mask)
    
    # Find end (last sound) - looking backwards
    end_idx = len(wav_np) - np.argmax(mask[::-1])
    
    # Add a safety buffer (50ms) so we don't clip the last syllable
    padding = int(0.05 * sample_rate)
    start_idx = max(0, start_idx - padding)
    end_idx = min(len(wav_np), end_idx + padding)
    
    return wav_np[start_idx:end_idx]

def crossfade_chunks(chunks_np, sample_rate=DEFAULT_SAMPLE_RATE, crossfade_ms=30, pause_ms=200):
    """
    THE FIX: Stitches chunks with a pause (for pacing) and micro-fades 
    to prevent clicks/pops at the join points.
    """
    if not chunks_np:
        return np.array([], dtype=np.float32)
    if len(chunks_np) == 1:
        return chunks_np[0]
        
    # Create silence buffer for the pause
    pause_len = int(sample_rate * (pause_ms / 1000))
    pause_silence = np.zeros(pause_len, dtype=np.float32)
    
    final_audio = chunks_np[0]
    
    for i in range(1, len(chunks_np)):
        next_chunk = chunks_np[i]
        
        # 1. Add the pause
        final_audio = np.concatenate([final_audio, pause_silence])
        
        # 2. Micro-fade to prevent pops
        # We fade out the last few ms of the previous clip and fade in the next
        fade_samples = 100 # Short fade just for zero-crossing safety
        
        # Fade out end
        if len(final_audio) > fade_samples:
            final_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
        # Fade in start
        if len(next_chunk) > fade_samples:
            next_chunk_faded = next_chunk.copy()
            next_chunk_faded[:fade_samples] *= np.linspace(0, 1, fade_samples)
        else:
            next_chunk_faded = next_chunk

        final_audio = np.concatenate([final_audio, next_chunk_faded])
        
    return final_audio

# --- 7. Generation Logic ---

def safe_generate(tts_model, gen_params):
    """Safely call generate, handling param mismatches between versions."""
    try:
        return tts_model.generate(**gen_params)
    except TypeError as e:
        # Fallback: remove advanced params if the installed model is old
        print(f"[generate] TypeError: {e}. Retrying with basic params.")
        p = dict(gen_params)
        for k in ("repetition_penalty", "top_p", "min_p"):
            p.pop(k, None)
        return tts_model.generate(**p)

def generate_chunk_with_guard(tts_model, base_params, chunk_text, base_seed, spc_threshold=800):
    """
    Generate one chunk. If the output is suspiciously short (premature stop),
    it retries with different parameters.
    """
    def one_pass(rp=None, top_p=None, min_p=None, seed_offset=0):
        p = dict(base_params)
        p["text"] = chunk_text
        if rp is not None: p["repetition_penalty"] = rp
        if top_p is not None: p["top_p"] = top_p
        if min_p is not None: p["min_p"] = min_p

        if base_seed:
            s = (base_seed + seed_offset) & 0xFFFFFFFF
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            np.random.seed(s)
            random.seed(s)

        out = safe_generate(tts_model, p)
        wav = to_numpy_1d(out)
        
        # Samples per character check
        spc = len(wav) / max(1, len(chunk_text))
        return wav, spc

    # Initial Attempt
    rp0 = float(base_params.get("repetition_penalty", 1.4))
    top_p0 = float(base_params.get("top_p", 0.9))
    min_p0 = float(base_params.get("min_p", 0.05))

    wav, spc = one_pass(rp=rp0, top_p=top_p0, min_p=min_p0, seed_offset=0)

    # Guard Check
    if spc < spc_threshold:
        print(f"[guard] Premature stop detected (spc={spc:.0f}). Retrying chunk...")
        # Retry with higher penalty and different seed
        wav, spc = one_pass(rp=max(1.1, rp0 + 0.2), top_p=top_p0, min_p=min_p0, seed_offset=1337)

    return wav

# --- 8. API Handlers ---

def clone_voice_handler(job):
    """
    Handles converting mobile audio uploads (m4a/mp4) to WAV and saving them.
    """
    try:
        job_input = job.get("input", {})
        audio_base64 = job_input.get("audio_data")
        voice_name = job_input.get("voice_name", "voice")
        user_id = job_input.get("user_id", "anonymous")

        print(f"[clone] Request received for user: {user_id}, voice: {voice_name}")

        if not audio_base64:
            return {"error": "audio_data is required"}

        user_dir = get_user_voice_dir(user_id)

        # 1. Decode Base64 to Temp File
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            return {"error": f"Failed to decode base64: {e}"}

        temp_filename = f"temp_upload_{uuid.uuid4().hex}.bin"
        temp_path = user_dir / temp_filename
        
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        # 2. Convert using FFmpeg (Critical for mobile uploads)
        # We convert whatever they sent -> 24000Hz Mono WAV
        final_filename = f"{voice_name}_{uuid.uuid4().hex[:6]}.wav"
        final_path = user_dir / final_filename

        print(f"[clone] Converting uploaded audio to {final_path}...")

        try:
            subprocess.run([
                "ffmpeg", "-y",          # Overwrite output
                "-i", str(temp_path),    # Input file
                "-ar", "24000",          # Sample rate expected by Chatterbox
                "-ac", "1",              # Mono
                str(final_path)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # Cleanup and fail
            if temp_path.exists(): os.remove(temp_path)
            return {"error": "FFmpeg conversion failed. Input format may be invalid."}
        
        # Cleanup temp file
        if temp_path.exists():
            os.remove(temp_path)

        print(f"[clone] Success. Saved to {final_filename}")

        return {
            "status": "success",
            "user_id": user_id,
            "embedding_filename": final_filename,
            "voice_name": voice_name,
            "message": "Voice cloned and normalized successfully"
        }

    except Exception as e:
        print(f"[error] Clone failed: {e}")
        traceback.print_exc()
        return {"error": f"Voice cloning failed: {str(e)}"}

def generate_tts_handler(job):
    """
    Main TTS Handler.
    """
    try:
        job_input = job.get("input", {})
        
        # --- Inputs ---
        text = job_input.get("text")
        preset_voice = job_input.get("preset_voice")
        user_id = job_input.get("user_id")
        embedding_filename = job_input.get("embedding_filename")
        
        # Settings
        language = job_input.get("language", "en")
        exaggeration = float(job_input.get("exaggeration", 0.6))
        temperature = float(job_input.get("temperature", 0.1))
        seed_input = int(job_input.get("seed", 0))

        print(f"[tts] Processing request for user: {user_id}")
        print(f"[tts] Exaggeration: {exaggeration}, Temp: {temperature}")

        if not text:
            return {"error": "text is required"}

        # --- 1. Resolve Audio Prompt ---
        
        # If preset provided, try to infer language from path (e.g. "es/Male1")
        if preset_voice and "/" in preset_voice:
            parts = preset_voice.split("/")
            if len(parts[0]) == 2:
                language = parts[0]
                print(f"[tts] Inferring language '{language}' from preset path")

        audio_path = get_voice_audio_path(
            user_id=user_id,
            embedding_filename=embedding_filename,
            preset_name=preset_voice
        )
        
        if not audio_path:
            print(f"[tts] Error: No valid voice file found.")
            return {"error": "Voice file not found (neither preset nor user voice)"}
        
        audio_prompt_path = str(audio_path)

        # --- 2. Preprocess Text ---
        text = clean_text_for_tts(text)
        
        # Chunking Logic
        MAX_CHARS = 400
        if len(text) > 500:
            print(f"[tts] Text is long ({len(text)} chars). Splitting...")
            sentences = split_sentences_safely(text)
            chunks = pack_sentences_into_chunks(sentences, max_chars=MAX_CHARS)
            # Add story pauses
            chunks = insert_quote_pauses(chunks)
            print(f"[tts] Created {len(chunks)} chunks.")
        else:
            chunks = [text]

        # --- 3. Load Model & Prepare Params ---
        tts_model = load_model()

        base_params = {
            "language_id": language,
            "audio_prompt_path": audio_prompt_path,
            "cfg_weight": float(job_input.get("cfg_weight", 0.3)),
            "exaggeration": exaggeration,
            "temperature": temperature,
            "repetition_penalty": float(job_input.get("repetition_penalty", 1.4)),
            "top_p": float(job_input.get("top_p", 0.9)),
            "min_p": float(job_input.get("min_p", 0.05)),
        }

        # --- 4. Generate Chunks ---
        all_chunks_audio_np = []

        for chunk_idx, chunk_text in enumerate(chunks):
            # Remove pause tags for generation, we handle timing via silence injection
            clean_chunk_text = chunk_text.replace("[pause]", "").strip()
            if not clean_chunk_text:
                continue

            print(f"[tts] Generating chunk {chunk_idx + 1}/{len(chunks)}...")
            
            per_chunk_seed = derive_seed(seed_input, chunk_idx) if seed_input else 0
            
            # Generate raw
            wav_np = generate_chunk_with_guard(
                tts_model, base_params, clean_chunk_text, per_chunk_seed,
                spc_threshold=int(job_input.get("spc_threshold", 800))
            )

            # FIX: Clean artifacts immediately
            wav_cleaned = trim_silence_and_artifacts(wav_np, threshold_db=-35)
            
            if len(wav_cleaned) > 0:
                all_chunks_audio_np.append(wav_cleaned)

        if not all_chunks_audio_np:
            raise RuntimeError("No audio chunks were generated")

        # --- 5. Stitching (The Fix) ---
        print(f"[tts] Stitching {len(all_chunks_audio_np)} chunks with crossfade...")
        
        final_wav = crossfade_chunks(
            all_chunks_audio_np, 
            sample_rate=DEFAULT_SAMPLE_RATE, 
            crossfade_ms=20, 
            pause_ms=200  # Storyteller pause duration
        )

        # Final Polish
        final_wav = gentle_hp(final_wav, cutoff=40.0)

        # Normalize Volume
        max_amp = np.max(np.abs(final_wav)) + 1e-12
        if max_amp > 0.99:
            final_wav = final_wav * (0.98 / max_amp)
            print(f"[tts] Normalized audio (Peak was {max_amp:.2f})")

        # --- 6. Encode Response ---
        buf = io.BytesIO()
        sf.write(buf, final_wav, DEFAULT_SAMPLE_RATE, format="WAV")
        buf.seek(0)
        audio_base64 = base64.b64encode(buf.read()).decode('utf-8')

        duration_sec = len(final_wav) / DEFAULT_SAMPLE_RATE
        print(f"[tts] Done. Duration: {duration_sec:.2f}s")

        return {
            "status": "success",
            "audio": audio_base64,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "format": "wav",
            "duration_seconds": round(duration_sec, 2),
        }

    except Exception as e:
        print(f"[error] TTS generation failed: {e}")
        traceback.print_exc()
        return {"error": f"TTS generation failed: {str(e)}"}

def list_preset_voices_handler(job):
    """
    Lists all available preset voices, grouped by language.
    """
    try:
        job_input = job.get("input", {})
        target_language = job_input.get("language")

        print(f"[presets] Listing voices (Filter: {target_language or 'None'})...")
        
        voices = []
        
        # Walk through the preset directory
        for path in PRESET_VOICES_ROOT.rglob("*"):
            if path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                # Get relative path (e.g., "en/Male_1.wav")
                rel_path = path.relative_to(PRESET_VOICES_ROOT)
                str_rel = str(rel_path)
                
                # Attempt language detection from folder structure
                lang = "unknown"
                parts = str_rel.split("/")
                if len(parts) > 1:
                    possible_lang = parts[0]
                    if len(possible_lang) == 2: # e.g. 'en', 'es'
                        lang = possible_lang
                
                if target_language and lang != "unknown" and lang != target_language:
                    continue

                voices.append({
                    "filename": str_rel,  # This is what the frontend sends back
                    "name": path.stem.replace("_", " ").title(),
                    "language": lang,
                    "size_bytes": path.stat().st_size
                })
        
        # Sort by language, then name
        voices.sort(key=lambda x: (x['language'], x['name']))
        
        print(f"[presets] Found {len(voices)} voices.")
        return {
            "status": "success", 
            "preset_voices": voices,
            "count": len(voices)
        }
    except Exception as e:
        print(f"[error] List presets failed: {e}")
        return {"error": str(e)}

def list_user_voices_handler(job):
    """
    Lists voices cloned by a specific user.
    """
    try:
        job_input = job.get("input", {})
        user_id = job_input.get("user_id")

        if not user_id:
            return {"error": "user_id is required"}

        user_dir = get_user_voice_dir(user_id)
        print(f"[user_voices] Listing for user: {user_id}")

        if not user_dir.exists():
            return {"status": "success", "user_voices": [], "count": 0}

        voices = []
        for f in user_dir.iterdir():
            if f.suffix.lower() in ['.wav', '.mp3']:
                # Format name nicely
                display_name = f.stem
                if "_" in display_name:
                    # remove the uuid part if possible for display
                    parts = display_name.split("_")
                    if len(parts) > 1:
                        display_name = " ".join(parts[:-1]) # Remove last part (uuid)
                
                voices.append({
                    "filename": f.name, # Critical for API
                    "name": display_name.title(),
                    "created_at": f.stat().st_mtime,
                    "size_bytes": f.stat().st_size
                })

        # Sort by newest first
        voices.sort(key=lambda x: x['created_at'], reverse=True)
        
        print(f"[user_voices] Found {len(voices)} voices.")
        return {
            "status": "success", 
            "user_voices": voices,
            "count": len(voices)
        }

    except Exception as e:
        print(f"[error] List user voices failed: {e}")
        return {"error": str(e)}

# --- 9. Main Entry Point ---

def handler(job):
    """
    Main router for RunPod requests.
    """
    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "generate_tts")

        # Dispatch
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
        print(f"[fatal] Unhandled exception in handler: {e}")
        traceback.print_exc()
        return {"error": f"Handler crashed: {str(e)}"}

if __name__ == "__main__":
    print("[startup] RunPod Handler starting...")
    print("[startup] Ready to accept requests.")
    runpod.serverless.start({"handler": handler})