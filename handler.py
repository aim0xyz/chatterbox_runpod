import runpod
import os
import torch
import base64
import json
import io
import soundfile as sf
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = Path("/qwen3_models")
VOICE_ROOT = Path("/runpod-volume/user_voices")
PRESET_ROOT = Path("/runpod-volume/preset_voices")

# Global model variable
model = None

# ==========================================
# SETUP & LOADING
# ==========================================
def init_model():
    """Load the Qwen3-TTS model."""
    global model
    print("[startup] Loading Qwen3-TTS model from local directory...")
    
    try:
        from qwen_tts import Qwen3TTSModel
        
        # Load model using the official Qwen3TTSModel wrapper
        # We pass dtype=torch.bfloat16 for efficiency on modern GPUs
        model = Qwen3TTSModel.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.bfloat16,
            device_map="auto"
        )
        print("[startup] Model loaded successfully!")
        
    except Exception as e:
        print(f"[startup] Error loading model: {e}")
        model = None

# Initialize on start
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
                return p
        
        # Fallback: search recursively
        for f in PRESET_ROOT.rglob(f"*{preset}*"):
            if f.is_file():
                return f
            
    # 2. User Voice
    if user_id and voice_id:
        p = get_user_dir(user_id) / voice_id
        # Try common extensions if voice_id doesn't have one
        if not p.suffix:
            for ext in ['.wav', '.mp3']:
                if (p.with_suffix(ext)).exists():
                    return p.with_suffix(ext)
        elif p.exists():
            return p
                
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
    language = inp.get("language", "en") # 'en', 'de', etc.
    user_id = inp.get("user_id")
    preset_voice = inp.get("preset_voice")
    voice_id = inp.get("voice_id") or inp.get("embedding_filename")
    
    # Generation parameters
    temperature = float(inp.get("temperature", 0.7))
    top_p = float(inp.get("top_p", 0.8))
    
    if not text:
        return {"error": "No text provided"}
        
    # Find prompt audio for zero-shot cloning
    voice_path = find_voice(user_id=user_id, voice_id=voice_id, preset=preset_voice)
    if not voice_path:
        return {"error": f"Voice not found: {preset_voice or voice_id}"}
        
    print(f"[TTS] Generating for user={user_id}, lang={language}, voice={voice_path} (temp={temperature}, top_p={top_p})")
    
    try:
        if not model:
             return {"error": "Model not loaded"}
             
        # CALL QWEN3-TTS
        # Official signature: model.generate_voice_clone(text, language, ref_audio, ...)
        # We use x_vector_only_mode=True as we don't always have the reference text for prompts
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(voice_path),
            x_vector_only_mode=True,
            temperature=temperature,
            top_p=top_p
        )
        
        # Qwen3 returns a list of numpy arrays, pick the first one
        audio_array = wavs[0]
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        buffer.seek(0)
        b64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {"audio": b64_audio}
        
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
    
    if not user_id or not voice_name or not audio_data:
        return {"error": "Missing user_id, voice_name, or audio_data"}
        
    # Destination: /runpod-volume/user_voices/{user_id}/{voice_name}.wav
    # Clean filename
    safe_name = "".join(x for x in voice_name if x.isalnum() or x in "-_")
    dest_path = get_user_dir(user_id) / f"{safe_name}.wav"
    
    print(f"[Clone] Saving voice '{voice_name}' for user {user_id} (path: {dest_path})")
    
    if save_base64_audio(audio_data, dest_path):
        return {
            "status": "success", 
            "voice_id": safe_name,
            "voice_name": voice_name,
            "user_id": user_id,
            "embedding_filename": f"{safe_name}.wav",
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
