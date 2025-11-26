"""
RunPod Serverless Handler for Chatterbox TTS - Storyteller Edition
Simplified and more robust version focused on not losing content.
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
    """Deterministically derive a seed for each chunk."""
    if not base_seed:
        return 0
    return (base_seed + (chunk_idx * 10007)) & 0xFFFFFFFF

def get_user_voice_dir(user_id: str) -> Path:
    """Return a sanitized, user-specific directory path."""
    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ("_", "-"))
    user_dir = VOICE_EMBEDDING_ROOT / safe_user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def get_voice_audio_path(user_id: str = None, embedding_filename: str = None, preset_name: str = None) -> Path:
    """Smart resolver: Checks Presets first, then User ID folders."""
    if preset_name:
        preset_path = PRESET_VOICES_ROOT / preset_name
        if preset_path.exists():
            print(f"[voice] Using preset voice: {preset_name}")
            return preset_path
        
        for f in PRESET_VOICES_ROOT.rglob(preset_name):
            print(f"[voice] Found fuzzy preset match: {f}")
            return f
            
    if user_id and embedding_filename:
        user_dir = get_user_voice_dir(user_id)
        audio_path = user_dir / embedding_filename
        if audio_path.exists():
            print(f"[voice] Using user voice: {user_id}/{embedding_filename}")
            return audio_path
        else:
            print(f"[warn] User voice file not found: {audio_path}")

    return None

# --- 5. Helper Functions: Text Processing (SIMPLIFIED) ---

def clean_text_for_tts(text: str) -> str:
    """Normalize quotes and remove invisible characters."""
    replacements = {
        "\u201c": '"', "\u201d": '"', 
        "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": " - ", 
        "\u2026": "...", "«": '"', "»": '"',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_chunks(text: str, max_chars: int = 300) -> list:
    """
    SIMPLIFIED chunking - split on paragraph breaks and sentence endings.
    Keeps dialogue intact by not splitting inside quotes.
    """
    if not text or not text.strip():
        return []
    
    # First, split on paragraph breaks (double newlines or explicit breaks)
    paragraphs = re.split(r'\n\s*\n', text)
    
    all_chunks = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If paragraph is short enough, keep it whole
        if len(para) <= max_chars:
            all_chunks.append(para)
            continue
        
        # Otherwise, split on sentence boundaries
        # This regex splits after .!? but NOT if inside quotes
        sentences = []
        current = ""
        in_quote = False
        
        i = 0
        while i < len(para):
            char = para[i]
            current += char
            
            # Track quote state
            if char in '""\'"':
                in_quote = not in_quote
            
            # Check for sentence end (only if not in quote)
            if char in '.!?' and not in_quote:
                # Look ahead to see if this is really the end
                next_char = para[i+1] if i+1 < len(para) else ' '
                if next_char in ' \n""\'' or i+1 >= len(para):
                    sentences.append(current.strip())
                    current = ""
            
            i += 1
        
        # Don't forget the last bit
        if current.strip():
            sentences.append(current.strip())
        
        # Now pack sentences into chunks
        current_chunk = ""
        for sent in sentences:
            if not current_chunk:
                current_chunk = sent
            elif len(current_chunk) + len(sent) + 1 <= max_chars:
                current_chunk += " " + sent
            else:
                all_chunks.append(current_chunk)
                current_chunk = sent
        
        if current_chunk:
            all_chunks.append(current_chunk)
    
    # Final validation
    final_chunks = [c.strip() for c in all_chunks if c.strip()]
    
    return final_chunks

# --- 6. Helper Functions: Audio Processing (CONSERVATIVE) ---

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
    """High-pass filter to remove sub-bass rumble."""
    try:
        from scipy import signal
        wav = wav_np.astype(np.float32)
        sos = signal.butter(2, cutoff, 'hp', fs=sample_rate, output='sos')
        return signal.sosfilt(sos, wav).astype(np.float32)
    except ImportError:
        return wav_np.astype(np.float32)

def trim_silence_only(wav_np, threshold_db=-45, sample_rate=DEFAULT_SAMPLE_RATE):
    """
    VERY CONSERVATIVE trimming - only removes true silence from edges.
    Uses -45dB threshold (very quiet) to avoid cutting speech.
    """
    if len(wav_np) == 0:
        return wav_np

    threshold_amp = 10 ** (threshold_db / 20)
    abs_wav = np.abs(wav_np)
    mask = abs_wav > threshold_amp

    if not np.any(mask):
        # Entirely silent - return minimal silence
        return np.zeros(int(0.05 * sample_rate), dtype=np.float32)

    # Find first and last sound
    start_idx = np.argmax(mask)
    end_idx = len(wav_np) - np.argmax(mask[::-1])

    # Generous padding (150ms) - storytelling needs room to breathe
    padding = int(0.15 * sample_rate)
    start_idx = max(0, start_idx - padding)
    end_idx = min(len(wav_np), end_idx + padding)

    return wav_np[start_idx:end_idx]

def apply_fade(wav_np, fade_in_ms=10, fade_out_ms=10, sample_rate=DEFAULT_SAMPLE_RATE):
    """Apply gentle fade in/out to prevent clicks."""
    if len(wav_np) == 0:
        return wav_np
    
    wav = wav_np.copy()
    
    fade_in_samples = min(int(fade_in_ms * sample_rate / 1000), len(wav) // 4)
    fade_out_samples = min(int(fade_out_ms * sample_rate / 1000), len(wav) // 4)
    
    if fade_in_samples > 0:
        wav[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples).astype(np.float32)
    
    if fade_out_samples > 0:
        wav[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples).astype(np.float32)
    
    return wav

def stitch_chunks(chunks_np, sample_rate=DEFAULT_SAMPLE_RATE, pause_ms=250):
    """
    Simple stitching with pauses between chunks.
    Uses 250ms pause for storytelling rhythm.
    """
    if not chunks_np:
        return np.array([], dtype=np.float32)
    
    if len(chunks_np) == 1:
        return apply_fade(chunks_np[0])
    
    pause_samples = int(sample_rate * pause_ms / 1000)
    pause = np.zeros(pause_samples, dtype=np.float32)
    
    result = apply_fade(chunks_np[0])
    
    for i in range(1, len(chunks_np)):
        chunk = apply_fade(chunks_np[i])
        result = np.concatenate([result, pause, chunk])
    
    return result

# --- 7. Generation Logic (SIMPLIFIED - NO GUARD) ---

def safe_generate(tts_model, gen_params):
    """Safely call generate, handling param mismatches."""
    try:
        return tts_model.generate(**gen_params)
    except TypeError as e:
        print(f"[generate] TypeError: {e}. Retrying with basic params.")
        p = dict(gen_params)
        for k in ("repetition_penalty", "top_p", "min_p"):
            p.pop(k, None)
        return tts_model.generate(**p)

def generate_chunk(tts_model, params, chunk_text, seed=0):
    """
    Generate audio for a single chunk.
    NO GUARD/RETRY - just generate once and trust the model.
    Retrying often makes things worse.
    """
    gen_params = dict(params)
    gen_params["text"] = chunk_text
    
    if seed:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    output = safe_generate(tts_model, gen_params)
    wav = to_numpy_1d(output)
    
    return wav

# --- 8. API Handlers ---

def clone_voice_handler(job):
    """Handles converting mobile audio uploads to WAV."""
    try:
        job_input = job.get("input", {})
        audio_base64 = job_input.get("audio_data")
        voice_name = job_input.get("voice_name", "voice")
        user_id = job_input.get("user_id", "anonymous")

        print(f"[clone] Request received for user: {user_id}, voice: {voice_name}")

        if not audio_base64:
            return {"error": "audio_data is required"}

        user_dir = get_user_voice_dir(user_id)

        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            return {"error": f"Failed to decode base64: {e}"}

        temp_filename = f"temp_upload_{uuid.uuid4().hex}.bin"
        temp_path = user_dir / temp_filename
        
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        final_filename = f"{voice_name}_{uuid.uuid4().hex[:6]}.wav"
        final_path = user_dir / final_filename

        print(f"[clone] Converting uploaded audio to {final_path}...")

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(temp_path),
                "-ar", "24000",
                "-ac", "1",
                str(final_path)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            if temp_path.exists(): os.remove(temp_path)
            return {"error": "FFmpeg conversion failed. Input format may be invalid."}
        
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
    """Main TTS Handler - Storyteller optimized."""
    try:
        job_input = job.get("input", {})
        
        # --- Inputs ---
        text = job_input.get("text")
        preset_voice = job_input.get("preset_voice")
        user_id = job_input.get("user_id")
        embedding_filename = job_input.get("embedding_filename")
        
        # Settings - Conservative defaults for storytelling
        language = job_input.get("language", "en")
        exaggeration = float(job_input.get("exaggeration", 0.5))  # Moderate expression
        temperature = float(job_input.get("temperature", 0.5))    # Stable output
        seed_input = int(job_input.get("seed", 0))
        
        # Chunking settings
        max_chunk_chars = int(job_input.get("max_chunk_chars", 300))
        pause_between_chunks_ms = int(job_input.get("pause_ms", 250))

        print(f"[tts] Request for user: {user_id}")
        print(f"[tts] Text length: {len(text) if text else 0} chars")
        print(f"[tts] Settings: exag={exaggeration}, temp={temperature}, max_chunk={max_chunk_chars}")

        if not text:
            return {"error": "text is required"}

        # --- 1. Resolve Voice ---
        if preset_voice and "/" in preset_voice:
            parts = preset_voice.split("/")
            if len(parts[0]) == 2:
                language = parts[0]
                print(f"[tts] Inferred language '{language}' from preset path")

        audio_path = get_voice_audio_path(
            user_id=user_id,
            embedding_filename=embedding_filename,
            preset_name=preset_voice
        )
        
        if not audio_path:
            return {"error": "Voice file not found"}
        
        audio_prompt_path = str(audio_path)

        # --- 2. Clean and Chunk Text ---
        text = clean_text_for_tts(text)
        
        if len(text) > max_chunk_chars:
            chunks = split_into_chunks(text, max_chars=max_chunk_chars)
            print(f"[tts] Split into {len(chunks)} chunks:")
            for i, c in enumerate(chunks):
                print(f"[tts]   {i+1}. ({len(c)} chars) \"{c[:60]}{'...' if len(c) > 60 else ''}\"")
        else:
            chunks = [text]
            print(f"[tts] Single chunk: {len(text)} chars")

        if not chunks:
            return {"error": "No valid text after processing"}

        # --- 3. Load Model ---
        tts_model = load_model()

        # Storyteller-optimized parameters
        gen_params = {
            "language_id": language,
            "audio_prompt_path": audio_prompt_path,
            "cfg_weight": float(job_input.get("cfg_weight", 0.5)),      # Lower = more natural
            "exaggeration": exaggeration,
            "temperature": temperature,
            "repetition_penalty": float(job_input.get("repetition_penalty", 1.05)),  # Very light
            "top_p": float(job_input.get("top_p", 0.95)),
            "min_p": float(job_input.get("min_p", 0.02)),
        }

        # --- 4. Generate Each Chunk ---
        audio_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                print(f"[tts] Skipping empty chunk {i+1}")
                continue

            print(f"[tts] Generating chunk {i+1}/{len(chunks)}...")
            
            seed = derive_seed(seed_input, i) if seed_input else 0
            
            try:
                wav = generate_chunk(tts_model, gen_params, chunk_text, seed)
                
                # Light cleanup only
                wav = trim_silence_only(wav, threshold_db=-45)
                
                duration_ms = len(wav) / DEFAULT_SAMPLE_RATE * 1000
                print(f"[tts]   ✓ Chunk {i+1}: {len(wav)} samples ({duration_ms:.0f}ms)")
                
                # Only skip if literally empty
                if len(wav) > 0:
                    audio_chunks.append(wav)
                else:
                    print(f"[tts]   ⚠ Chunk {i+1} produced no audio")
                    
            except Exception as e:
                print(f"[tts]   ✗ Chunk {i+1} failed: {e}")
                traceback.print_exc()
                # Continue with other chunks instead of failing entirely

        if not audio_chunks:
            return {"error": "No audio was generated"}

        print(f"[tts] Generated {len(audio_chunks)}/{len(chunks)} chunks successfully")

        # --- 5. Stitch Together ---
        final_wav = stitch_chunks(
            audio_chunks, 
            sample_rate=DEFAULT_SAMPLE_RATE, 
            pause_ms=pause_between_chunks_ms
        )

        # Light filtering
        final_wav = gentle_hp(final_wav, cutoff=35.0)

        # Normalize if needed
        max_amp = np.max(np.abs(final_wav))
        if max_amp > 0.95:
            final_wav = final_wav * (0.95 / max_amp)

        # --- 6. Encode ---
        buf = io.BytesIO()
        sf.write(buf, final_wav, DEFAULT_SAMPLE_RATE, format="WAV")
        buf.seek(0)
        audio_base64 = base64.b64encode(buf.read()).decode('utf-8')

        duration_sec = len(final_wav) / DEFAULT_SAMPLE_RATE
        print(f"[tts] ✅ Complete. Duration: {duration_sec:.2f}s")

        return {
            "status": "success",
            "audio": audio_base64,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "format": "wav",
            "duration_seconds": round(duration_sec, 2),
            "chunks_generated": len(audio_chunks),
            "chunks_requested": len(chunks),
        }

    except Exception as e:
        print(f"[error] TTS generation failed: {e}")
        traceback.print_exc()
        return {"error": f"TTS generation failed: {str(e)}"}

def list_preset_voices_handler(job):
    """Lists all available preset voices."""
    try:
        job_input = job.get("input", {})
        target_language = job_input.get("language")

        print(f"[presets] Listing voices (Filter: {target_language or 'None'})...")
        
        voices = []
        
        for path in PRESET_VOICES_ROOT.rglob("*"):
            if path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                rel_path = path.relative_to(PRESET_VOICES_ROOT)
                str_rel = str(rel_path)
                
                lang = "unknown"
                parts = str_rel.split("/")
                if len(parts) > 1:
                    possible_lang = parts[0]
                    if len(possible_lang) == 2:
                        lang = possible_lang
                
                if target_language and lang != "unknown" and lang != target_language:
                    continue

                voices.append({
                    "filename": str_rel,
                    "name": path.stem.replace("_", " ").title(),
                    "language": lang,
                    "size_bytes": path.stat().st_size
                })
        
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
    """Lists voices cloned by a specific user."""
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
                display_name = f.stem
                if "_" in display_name:
                    parts = display_name.split("_")
                    if len(parts) > 1:
                        display_name = " ".join(parts[:-1])
                
                voices.append({
                    "filename": f.name,
                    "name": display_name.title(),
                    "created_at": f.stat().st_mtime,
                    "size_bytes": f.stat().st_size
                })

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
    """Main router for RunPod requests."""
    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "generate_tts")

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
        print(f"[fatal] Unhandled exception: {e}")
        traceback.print_exc()
        return {"error": f"Handler crashed: {str(e)}"}

if __name__ == "__main__":
    print("[startup] RunPod Handler (Storyteller Edition) starting...")
    runpod.serverless.start({"handler": handler})