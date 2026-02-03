import runpod
import os
import torch
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
    """Load the Qwen3-TTS model."""
    global model
    try:
        from qwen_tts import Qwen3TTSModel
        print(f"[startup] Loading Qwen3-TTS model from {MODEL_PATH}...")
        model = Qwen3TTSModel.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure tokenizer padding to avoid tensor creation errors
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            model.tokenizer.padding_side = 'left'
            model.tokenizer.pad_token = model.tokenizer.eos_token
            
        print("[startup] Model loaded successfully!")
    except Exception as e:
        print(f"[startup] Error loading model: {e}")

# Call init
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
    
    # Generation parameters (optimized for voice fidelity + storytelling)
    # Lower temperature = more consistent voice matching
    # Higher temperature = more expressive but may drift from original voice
    temperature = float(inp.get("temperature", 0.65))  # Conservative for voice fidelity
    top_p = float(inp.get("top_p", 0.8))               # Natural variation
    repetition_penalty = float(inp.get("repetition_penalty", 1.05))  # Prevents monotony
    top_k = int(inp.get("top_k", 35))                  # Balanced token selection
    
    # High default for long stories (8192 tokens ≈ 10 minutes of reading)
    max_new_tokens = int(inp.get("max_new_tokens", 8192))
    
    if not text:
        return {"error": "No text provided"}
        
    # Find prompt audio for zero-shot cloning
    voice_path = find_voice(user_id=user_id, voice_id=voice_id, preset=preset_voice)
    if not voice_path:
        return {"error": f"Voice not found: {preset_voice or voice_id}"}
        
    print(f"[TTS] Generating for user={user_id}, lang={language} (from {lang_code}), voice={voice_path}")
    print(f"[TTS] Parameters: temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}, top_k={top_k}")
    
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
        
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(voice_path),
            x_vector_only_mode=True,
            **gen_kwargs  # Pass all generation parameters
        )
        
        # Qwen3 returns a list of numpy arrays, pick the first one
        audio_array = wavs[0]
        print(f"[TTS] Generated audio shape: {audio_array.shape}, sample_rate: {sample_rate}")
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        buffer.seek(0)
        b64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        print(f"[TTS] Encoded audio size: {len(b64_audio)} bytes")
        
        return {
            "status": "success",
            "audio": b64_audio,
            "sample_rate": sample_rate,
            "format": "wav"
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
