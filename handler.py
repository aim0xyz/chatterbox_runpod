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
            device_map="auto",
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
    """Handle text-to-speech request."""
    inp = job["input"]
    text = inp.get("text")
    lang_code = inp.get("language", "en") # 'en', 'de', etc.
    
    # Translate language code to full name (e.g. 'de' -> 'german')
    language = LANG_MAP.get(lang_code.lower(), "auto")
    
    user_id = inp.get("user_id")
    preset_voice = inp.get("preset_voice")
    voice_id = inp.get("voice_id") or inp.get("embedding_filename")
    
    # Generation parameters (balanced for stability and expression)
    # 0.85 temperature is the 'sweet spot' for stability without rushing
    # 1.05 repetition penalty prevents the model from speeding up or looping
    temperature = float(inp.get("temperature", 0.85))  # Balanced (was 0.95)
    top_p = float(inp.get("top_p", 0.92))              # Keep high for prosody variation
    repetition_penalty = float(inp.get("repetition_penalty", 1.05))  # Stabilizer (was 1.0)
    top_k = int(inp.get("top_k", 60))                  # Maximum voice variation (was 50)
    
    if not text:
        return {"error": "No text provided"}

    # --- STORYTELLER VIBE: Natural Pause Injection ---
    # To prevent 'rushing' between sentences, we add breathing room.
    # We replace standard punctuation with versions that trigger natural pauses.
    print(f"[storyteller] Enhancing text for better rhythm and pauses...")
    enhanced_text = text.replace(". ", "... ") \
                       .replace("! ", "!!! ") \
                       .replace("? ", "?? ") \
                       .replace("\n\n", ".\n\n[pause]\n\n") \
                       .strip()
    
    # Use the enhanced text for generation
    original_text = text
    text = enhanced_text
    
    # --- CONCLUDING INTONATION FIX ---
    # Add extra punctuation at the very end to signal a definitive wrap-up.
    # This forces the pitch to drop and adds a natural final silence.
    print(f"[storyteller] Adding concluding intonation...")
    if text.endswith(".") or text.endswith("!") or text.endswith("?"):
        text = text + "....  "
    else:
        text = text + "....  "

    
    # Smart token calculation (use original length for base, enhanced for limit)
    estimated_tokens = int(len(original_text) * 1.5)  # ~1.5 tokens per char
    buffer_tokens = int(estimated_tokens * 0.1)  # 10% buffer
    calculated_max_tokens = estimated_tokens + buffer_tokens
    
    # Override with manual setting if provided, otherwise use calculated value
    # Clamp between 512 and 8192 for safety
    max_new_tokens = int(inp.get("max_new_tokens", calculated_max_tokens))
    max_new_tokens = max(512, min(max_new_tokens, 8192))
    
    print(f"[TTS] Text length: {len(text)} chars")
    print(f"[TTS] Calculated max_new_tokens: {calculated_max_tokens} (using: {max_new_tokens})")
    
    # Find prompt audio for zero-shot cloning
    voice_path = find_voice(user_id=user_id, voice_id=voice_id, preset=preset_voice)
    if not voice_path:
        return {"error": f"Voice not found: {preset_voice or voice_id}"}
        
    print(f"\n{'='*60}")
    print(f"[TTS] 🎙️  STARTING TTS GENERATION")
    print(f"[TTS] User: {user_id}")
    print(f"[TTS] Language: {language} ({lang_code})")
    print(f"[TTS] Voice: {voice_path}")
    print(f"[TTS] Text length: {len(text)} chars")
    print(f"[TTS] Parameters: temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}, top_k={top_k}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    progress_log = []  # Track progress for frontend
    
    def log_progress(step: str, progress: int, message: str, **extra):
        """Helper to log progress both to console and tracking list"""
        elapsed = time.time() - start_time
        entry = {
            "step": step,
            "progress": progress,
            "message": message,
            "elapsed_seconds": round(elapsed, 2),
            **extra
        }
        progress_log.append(entry)
        print(f"[Progress] {progress}% - {message}")
    
    try:
        if not model:
             return {"error": "Model not loaded"}
             
        # CALL QWEN3-TTS
        # Official signature: model.generate_voice_clone(text, language, ref_audio, ...)
        # We use x_vector_only_mode=True as we don't always have the reference text for prompts
        
        # Build generation kwargs
        gen_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,  # Always set for long stories
        }
        
        log_progress("init", 0, "Initializing generation...")
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(voice_path),
            x_vector_only_mode=True,
            **gen_kwargs  # Pass all generation parameters
        )
        log_progress("generate", 90, "Audio generated!")
        
        # Qwen3 returns a list of numpy arrays, pick the first one
        audio_array = wavs[0]
        duration = len(audio_array) / sample_rate
        print(f"[TTS] Generated audio shape: {audio_array.shape}, sample_rate: {sample_rate}")
        print(f"[TTS] Audio duration: {duration:.2f} seconds")
        
        # Convert to MP3 for smaller file size (10x reduction vs WAV)
        # First write to WAV in memory
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_array, sample_rate, format='WAV')
        wav_buffer.seek(0)
        log_progress("convert", 92, "Converting to MP3...")
        
        # Convert WAV to MP3 using pydub
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_wav(wav_buffer)
        
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format='mp3', bitrate='128k')
        mp3_buffer.seek(0)
        log_progress("encode", 95, "Encoding to base64...")
        
        b64_audio = base64.b64encode(mp3_buffer.read()).decode('utf-8')
        total_time = time.time() - start_time
        throughput = duration / total_time if total_time > 0 else 0
        
        print(f"[TTS] Encoded MP3 size: {len(b64_audio)} bytes (WAV would be ~{len(b64_audio)*10} bytes)")
        print(f"\n{'='*60}")
        print(f"[Progress] ✅ 100% - COMPLETE! Total time: {total_time:.2f}s")
        print(f"[TTS] Throughput: {throughput:.2f}x realtime")
        print(f"{'='*60}\n")
        
        log_progress("complete", 100, "Complete!", 
                    total_time=round(total_time, 2),
                    audio_duration=round(duration, 2),
                    throughput=round(throughput, 2))
        
        return {
            "status": "success",
            "audio": b64_audio,
            "sample_rate": sample_rate,
            "format": "mp3",
            "progress_log": progress_log,  # Include progress timeline
            "stats": {
                "total_time_seconds": round(total_time, 2),
                "audio_duration_seconds": round(duration, 2),
                "throughput_realtime": round(throughput, 2),
                "file_size_bytes": len(b64_audio)
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
