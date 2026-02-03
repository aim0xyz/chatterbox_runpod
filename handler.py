import runpod
import os
import torch
import torch._dynamo
import base64
import json
import io
import time
import soundfile as sf
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
# Model and Voice Paths
MODEL_PATH = Path("/qwen3_models")
VOICE_ROOT = Path("/runpod-volume/user_voices")
PRESET_ROOT = Path("/runpod-volume/preset_voices")

# --- SMART PATH DISCOVERY ---
# 1. Try path discovered by start.sh
if Path("/tmp/model_path.txt").exists():
    with open("/tmp/model_path.txt", "r") as f:
        discovered = Path(f.read().strip())
        if (discovered / "model.safetensors").exists():
            MODEL_PATH = discovered
            print(f"[startup] Path set from search: {MODEL_PATH}")

# 2. Fallback recursive search if still not found
if not (MODEL_PATH / "model.safetensors").exists():
    print("[startup] Model not in default path, searching common root /runpod-volume...")
    for path in Path("/runpod-volume").rglob("model.safetensors"):
        if "speech_tokenizer" not in str(path):
            MODEL_PATH = path.parent
            print(f"[startup] Found model at fallback: {MODEL_PATH}")
            break

# Global model variable
model = None

# Language mapping (Qwen3 wants full names, not ISO codes)
LANG_MAP = {
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish"
}

def init_model():
    """Load the Qwen3-TTS model with singleton check."""
    global model
    if model is not None:
        print("[startup] Model already loaded, skipping...")
        return
        
    try:
        from qwen_tts import Qwen3TTSModel
        print(f"[startup] Loading Qwen3-TTS model from {MODEL_PATH}...")
        model = Qwen3TTSModel.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True
        )
        
        # Configure tokenizer padding
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            model.tokenizer.padding_side = 'left'
            model.tokenizer.pad_token = model.tokenizer.eos_token
        
        # NOTE: torch.compile() is disabled to prevent graph breaks
        print("[startup] Model loaded successfully into GPU!")
    except Exception as e:
        print(f"[startup] Error loading model: {e}")

# Call init once at startup
if __name__ == "__main__" or os.environ.get("RUNPOD_AGENT_ID"):
    init_model()

# ==========================================
# HELPERS
# ==========================================
def get_user_dir(user_id):
    safe = "".join(c for c in user_id if c.isalnum() or c in "_-")
    d = VOICE_ROOT / safe
    d.mkdir(parents=True, exist_ok=True)
    return d

def find_voice(user_id=None, voice_id=None, preset=None):
    """Locate the voice file (user or preset)."""
    print(f"[find_voice] Searching for: user_id={user_id}, voice_id={voice_id}, preset={preset}")
    
    # 1. Preset Voice
    if preset:
        # Try direct path variations first
        for ext in ['', '.wav', '.mp3', '.flac']:
            p = PRESET_ROOT / f"{preset}{ext}"
            if p.exists():
                print(f"[find_voice] ✅ Found preset at: {p}")
                return p
        
        # Fallback: search recursively in presets
        for f in PRESET_ROOT.rglob(f"*{preset}*"):
            if f.is_file():
                print(f"[find_voice] ✅ Found preset (recursive) at: {f}")
                return f
        
        # If not found in presets, try searching in user_voices
        # This handles cases where user voices are incorrectly sent as "preset"
        print(f"[find_voice] ⚠️ Not found in presets, searching user_voices...")
        for f in VOICE_ROOT.rglob(f"*{preset}*"):
            if f.is_file():
                print(f"[find_voice] ✅ Found user voice at: {f}")
                return f
            
    # 2. User Voice
    if user_id and voice_id:
        p = get_user_dir(user_id) / voice_id
        # Try common extensions if voice_id doesn't have one
        if not p.suffix:
            for ext in ['.wav', '.mp3']:
                if (p.with_suffix(ext)).exists():
                    print(f"[find_voice] ✅ Found user voice at: {p.with_suffix(ext)}")
                    return p.with_suffix(ext)
        elif p.exists():
            print(f"[find_voice] ✅ Found user voice at: {p}")
            return p
    
    print(f"[find_voice] ❌ Voice not found anywhere")
    return None

def save_base64_audio(b64_data, dest_path):
    """Save base64 audio string to file."""
    try:
        if "base64," in b64_data:
            b64_data = b64_data.split("base64,")[1]
        
        audio_bytes = base64.b64decode(b64_data)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        with open(dest_path, "wb") as f:
            f.write(audio_bytes)
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

# ==========================================
# HANDLERS
# ==========================================
def generate_tts_handler(job):
    """Handle text-to-speech request with smart chunking."""
    inp = job["input"]
    text = inp.get("text")
    lang_code = inp.get("language", "en") # 'en', 'de', etc.
    user_id = inp.get("user_id")
    preset_voice = inp.get("preset_voice")
    voice_id = inp.get("voice_id") or inp.get("embedding_filename")
    
    # Translate language code to full name (e.g. 'de' -> 'german')
    language = LANG_MAP.get(lang_code.lower(), "auto")
    
    # --- VOICE SETTINGS ---
    temperature = float(inp.get("temperature", 0.75)) 
    top_p = float(inp.get("top_p", 0.95))             
    repetition_penalty = float(inp.get("repetition_penalty", 1.05)) 
    top_k = int(inp.get("top_k", 75))
    
    if not text:
        return {"error": "No text provided"}

    # Find prompt audio
    voice_path = find_voice(user_id=user_id, voice_id=voice_id, preset=preset_voice)
    if not voice_path:
        return {"error": f"Voice not found: {preset_voice or voice_id}"}

    # --- THE STITCHER: SMART CHUNKING ---
    def split_into_chunks(t, max_chars=400):
        # Split by sentence markers but keep punctuation
        sentences = re.split('(?<=[.!?]) +', t)
        chunks = []
        current_chunk = ""
        for s in sentences:
            if len(current_chunk) + len(s) < max_chars:
                current_chunk += " " + s
            else:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = s
        if current_chunk: chunks.append(current_chunk.strip())
        return chunks

    text_chunks = split_into_chunks(text)
    
    print(f"\n{'='*60}")
    print(f"[TTS] 🎙️  STARTING CHUNKED GENERATION")
    print(f"[TTS] Processing {len(text_chunks)} chunks for {len(text)} chars...")
    print(f"[TTS] Language: {language} | Voice: {voice_path}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    all_audio = []
    sr = 24000
    progress_log = []
    
    def log_progress(step: str, progress: int, message: str, **extra):
        elapsed = time.time() - start_time
        entry = {"step": step, "progress": progress, "message": message, "elapsed_seconds": round(elapsed, 2), **extra}
        progress_log.append(entry)
        print(f"[Progress] {progress}% - {message}")

    try:
        if model is None:
             return {"error": "Model not loaded"}

        for i, chunk in enumerate(text_chunks):
            # Log progress for UI
            progress_pct = int((i / len(text_chunks)) * 90)
            log_progress("generate", progress_pct, f"Storyteller speaking (Chunk {i+1}/{len(text_chunks)})...")
            
            # Generate this specific chunk
            # Parameters optimized for high-quality cloning
            wavs, sample_rate = model.generate_voice_clone(
                text=chunk,
                language=language,
                ref_audio=str(voice_path),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                x_vector_only_mode=True
            )
            sr = sample_rate
            all_audio.append(wavs[0])
            
            # Add 400ms silence between sentences for natural storyteller rhythm
            silence = np.zeros(int(0.4 * sr))
            all_audio.append(silence)
            
        # Combine pieces
        final_audio = np.concatenate(all_audio)
        duration = len(final_audio) / sr
        
        # 3. Encoding to MP3
        log_progress("convert", 92, "Finishing story audio...")
        
        wav_io = io.BytesIO()
        sf.write(wav_io, final_audio, sr, format='WAV')
        wav_io.seek(0)
        
        audio_segment = AudioSegment.from_wav(wav_io)
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="192k")
        b64_audio = base64.b64encode(mp3_io.getvalue()).decode('utf-8')
        
        total_time = time.time() - start_time
        throughput = duration / total_time if total_time > 0 else 0
        
        log_progress("complete", 100, "Complete!", total_time=round(total_time, 2), duration=round(duration, 2))
        
        return {
            "status": "success",
            "audio": b64_audio,
            "sample_rate": sr,
            "format": "mp3",
            "progress_log": progress_log,
            "stats": {
                "total_time_seconds": round(total_time, 2),
                "audio_duration_seconds": round(duration, 2),
                "throughput_realtime": round(throughput, 2)
            }
        }
        
    except Exception as e:
        print(f"Generation error: {e}")
        return {"error": str(e)}

def clone_voice_handler(job):
    """
    Handle voice cloning (Upload & Register).
    For Zero-Shot models like Qwen, this simply involves saving the reference audio.
    """
    inp = job["input"]
    user_id = inp.get("user_id")
    voice_name = inp.get("voice_name")
    audio_data = inp.get("audio_data") # Base64
    
    # Create a unique filename with timestamp to avoid collisions
    # Format: {safe_name}_{timestamp}.wav
    # This matches the parsing logic in UserVoice.fromJson in the Flutter app
    timestamp = int(time.time())
    safe_name = "".join(x for x in voice_name if x.isalnum() or x in "-_")
    filename = f"{safe_name}_{timestamp}.wav"
    dest_path = get_user_dir(user_id) / filename
    
    print(f"[Clone] Saving voice '{voice_name}' for user {user_id} (path: {dest_path})")
    
    if save_base64_audio(audio_data, dest_path):
        return {
            "status": "success", 
            "voice_id": filename, # Return the full filename as the ID
            "voice_name": voice_name,
            "user_id": user_id,
            "embedding_filename": filename,
            "message": "Voice saved successfully for zero-shot inference"
        }
    else:
        return {"error": "Failed to save voice audio"}

# ==========================================
# MAIN DISPATCHER
# ==========================================
def handler(job):
    """Entry point for RunPod."""
    inp = job.get("input", {})
    action = inp.get("action", "generate_tts")
    
    print(f"[Handler] Request Action: {action}")
    
    if action == "generate_tts":
        return generate_tts_handler(job)
    elif action == "clone_voice":
        return clone_voice_handler(job)
    else:
        return {"error": f"Unknown action: {action}"}

# Start RunPod server
runpod.serverless.start({"handler": handler})
