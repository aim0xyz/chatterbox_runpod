import runpod
import os
import torch
import torch._dynamo
import base64
import json
import io
import time
import re
import soundfile as sf
import numpy as np
from pathlib import Path
from pydub import AudioSegment

# ==========================================
# CONFIGURATION
# ==========================================
# Model and Voice Paths
# DEFAULT TO 0.6B REPO FOR MAXIMUM SPEED (8s target)
MODEL_NAME_OR_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" 
VOICE_ROOT = Path("/runpod-volume/user_voices")
PRESET_ROOT = Path("/runpod-volume/preset_voices")

# --- SMART PATH DISCOVERY ---
# 1. PRIORITIZE the manual 0.6B folder (Check both volume and workspace)
SEARCH_PATHS = [Path("/workspace/qwen3_0.6b"), Path("/runpod-volume/qwen3_0.6b")]
MODEL_NAME_OR_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" # Default to repo

for p in SEARCH_PATHS:
    if (p / "model.safetensors").exists():
        MODEL_NAME_OR_PATH = str(p)
        print(f"[startup] SUCCESS: Using manual 0.6B Local Model: {MODEL_NAME_OR_PATH}")
        break

# 2. FALLBACK to the Image path or Repo
if MODEL_NAME_OR_PATH == "Qwen/Qwen3-TTS-12Hz-0.6B-Base":
    if (Path("/qwen3_models") / "model.safetensors").exists():
        MODEL_NAME_OR_PATH = "/qwen3_models"
        print(f"[startup] Using FAST 0.6B Image Model: {MODEL_NAME_OR_PATH}")
    else:
        print(f"[startup] No local match found, using Repo ID (will auto-download): {MODEL_NAME_OR_PATH}")

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
    """Load the Qwen3-TTS model with singleton check and warm-up."""
    global model
    if model is not None:
        return
        
    try:
        from qwen_tts import Qwen3TTSModel
        print(f"[startup] Loading Qwen3-TTS model: {MODEL_NAME_OR_PATH}...")
        
        # Speed Optimization: Enable TF32 for faster math on modern GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        model = Qwen3TTSModel.from_pretrained(
            str(MODEL_NAME_OR_PATH),
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        # Configure tokenizer
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            model.tokenizer.padding_side = 'left'
            model.tokenizer.pad_token = model.tokenizer.eos_token

        # --- FLASH ATTENTION VERIFICATION ---
        print(f"\n{'='*60}")
        print(f"[FA2 CHECK] 🔍 Verifying Flash Attention Status")
        print(f"{'='*60}")
        
        # Check 1: Is flash_attn installed?
        try:
            import flash_attn
            print(f"[FA2 CHECK] ✅ flash_attn module installed (v{flash_attn.__version__})")
        except ImportError:
            print(f"[FA2 CHECK] ❌ flash_attn module NOT installed")
        
        # Check 2: Inspect Qwen3TTSModel wrapper
        print(f"[FA2 CHECK] Model type: {type(model).__name__}")
        
        # Check 2.5: Verify GPU usage
        print(f"\n[FA2 CHECK] 🖥️  Device Check:")
        print(f"[FA2 CHECK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[FA2 CHECK] CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"[FA2 CHECK] Current device: cuda:{torch.cuda.current_device()}")
        
        # Check where model parameters are actually located
        cpu_params = 0
        gpu_params = 0
        try:
            for name, param in model.named_parameters():
                if param.device.type == 'cpu':
                    cpu_params += 1
                    if cpu_params == 1:  # Only print first one
                        print(f"[FA2 CHECK] ⚠️  Found CPU parameter: {name}")
                elif param.device.type == 'cuda':
                    gpu_params += 1
                    if gpu_params == 1:  # Only print first one
                        print(f"[FA2 CHECK] ✅ Found GPU parameter: {name} (device={param.device})")
        except:
            print(f"[FA2 CHECK] ⚠️  Could not iterate parameters")
        
        if cpu_params > 0:
            print(f"[FA2 CHECK] ❌ CRITICAL: {cpu_params} parameters on CPU!")
            print(f"[FA2 CHECK] 🐌 This WILL cause 10-20x slowdown!")
        elif gpu_params > 0:
            print(f"[FA2 CHECK] ✅ All {gpu_params} parameters on GPU")
        
        # Try to find the underlying transformer model inside the wrapper
        underlying_model = None
        if hasattr(model, 'model'):
            underlying_model = model.model
            print(f"[FA2 CHECK] Found underlying model at: model.model")
        elif hasattr(model, 'talker'):
            underlying_model = model.talker
            print(f"[FA2 CHECK] Found underlying model at: model.talker")
        
        # Check config of underlying model
        if underlying_model and hasattr(underlying_model, 'config'):
            attn_impl = getattr(underlying_model.config, '_attn_implementation', 'unknown')
            print(f"[FA2 CHECK] Underlying model attn_implementation: {attn_impl}")
            if attn_impl == 'flash_attention_2':
                print(f"[FA2 CHECK] ✅ Underlying model CONFIGURED for FA2")
            else:
                print(f"[FA2 CHECK] ❌ Underlying model NOT configured for FA2 (using: {attn_impl})")
        
        # Check 3: Look for Flash Attention in the model's submodules
        fa2_found = False
        standard_attn_found = False
        
        try:
            # Try to iterate through modules (works for standard PyTorch models)
            search_model = underlying_model if underlying_model else model
            
            if hasattr(search_model, 'named_modules'):
                for name, module in search_model.named_modules():
                    module_type = type(module).__name__
                    
                    if 'Flash' in module_type or 'flash' in module_type.lower():
                        fa2_found = True
                        print(f"[FA2 CHECK] ✅ Found FA2 layer: {name} ({module_type})")
                        break
                    
                    if 'Attention' in module_type and 'Flash' not in module_type and not standard_attn_found:
                        standard_attn_found = True
                        print(f"[FA2 CHECK] ⚠️  Found standard attention: {name} ({module_type})")
        except Exception as e:
            print(f"[FA2 CHECK] ⚠️  Could not inspect model layers: {e}")
        
        # Summary
        print(f"\n[FA2 CHECK] 📊 VERDICT:")
        if fa2_found:
            print(f"[FA2 CHECK] ✅ Flash Attention 2 IS ACTIVE")
            print(f"[FA2 CHECK] 🚀 Expected: 100+ tokens/sec on L4/A5000")
        else:
            print(f"[FA2 CHECK] ❌ Flash Attention 2 NOT DETECTED")
            print(f"[FA2 CHECK] 🐌 This explains 13-23 tokens/sec performance")
            print(f"[FA2 CHECK] 💡 qwen-tts may not support FA2 acceleration")
        
        print(f"{'='*60}\n")

        # --- THE WARM-UP ---
        # This forces the 'code_predictor' and 'speaker_encoder' to initialize NOW
        print("[startup] Warming up model engine...")
        dummy_ref = str(PRESET_ROOT / "en/Owen.wav") # Owen is reliably 24khz
        if not os.path.exists(dummy_ref):
            # Fallback to any file in presets
            for f in PRESET_ROOT.rglob("*.wav"):
                dummy_ref = str(f)
                break

        if os.path.exists(dummy_ref):
            try:
                with torch.inference_mode():
                    model.generate_voice_clone(
                        text="Hi.", 
                        language="english", 
                        ref_audio=dummy_ref,
                        x_vector_only_mode=True, # MUST match generation logic
                        max_new_tokens=5
                    )
                print("[startup] Warm-up complete! Engine is hot.")
            except Exception as e:
                print(f"[startup] Warm-up skipped/failed (non-critical): {e}")
        
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
    def split_into_chunks(t, max_chars=300):
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

        # --- SPEED INJECTOR 1: Voice DNA Caching ---
        print(f"[TTS] 🧬 Pre-encoding voice signature: {voice_path}")
        prep_start = time.time()
        
        # Load and clip reference audio to 5s if it's long (saves ~15s per chunk)
        import torchaudio
        try:
            waveform, sample_rate_ref = torchaudio.load(str(voice_path))
            if waveform.shape[1] > (5 * sample_rate_ref):
                print(f"[TTS] ✂️  Clipping long voice prompt to 5s for speed")
                waveform = waveform[:, :int(5 * sample_rate_ref)]
                temp_voice = "/tmp/speed_ref.wav"
                torchaudio.save(temp_voice, waveform, sample_rate_ref)
                voice_path = temp_voice
        except Exception as e:
            print(f"[TTS] ⚠️  Voice clipping skipped: {e}")

        # Ensure we use high-speed no-grad mode
        torch.set_grad_enabled(False)

        print(f"\n{'='*60}")
        print(f"[PROFILER] 🔍 DEEP PERFORMANCE ANALYSIS")
        print(f"{'='*60}\n")

        for i, chunk in enumerate(text_chunks):
            # Log progress for UI
            progress_pct = int((i / len(text_chunks)) * 90)
            log_progress("generate", progress_pct, f"Storyteller speaking (Chunk {i+1}/{len(text_chunks)})...")
            
            print(f"\n[PROFILER] --- Chunk {i+1}/{len(text_chunks)} ---")
            print(f"[PROFILER] Text: '{chunk[:50]}...' ({len(chunk)} chars)")
            
            chunk_start = time.time()
            # SPEED FIX 2: Hard limit on generation length
            limit = int(len(chunk) * 2.2) + 60
            print(f"[PROFILER] Token limit: {limit}")
            
            # Sync GPU before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            t0 = time.time()
            print(f"[PROFILER] ⏱️  Starting generation at t={0:.2f}s...")
            
            # Generate this specific chunk
            # Forced Zero-Shot Mode (No ICL) for maximum speed
            wavs, sample_rate = model.generate_voice_clone(
                text=chunk,
                language=language,
                ref_audio=str(voice_path),
                ref_text=None,
                x_vector_only_mode=True, 
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                max_new_tokens=limit
            )
            
            # Sync GPU after generation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            t1 = time.time()
            gen_time = t1 - t0
            
            chunk_time = time.time() - chunk_start
            audio_duration = len(wavs[0]) / sample_rate
            
            print(f"[PROFILER] ✅ Generation complete at t={chunk_time:.2f}s")
            print(f"[PROFILER] 📊 Breakdown:")
            print(f"[PROFILER]   - Pure generation: {gen_time:.2f}s")
            print(f"[PROFILER]   - Audio output: {audio_duration:.2f}s")
            print(f"[PROFILER]   - Real-time ratio: {audio_duration/gen_time:.2f}x")
            print(f"[PROFILER]   - Tokens/sec: {round(limit/gen_time, 1)}")
            
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[PROFILER]   - GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            
            print(f"[PROFILER] Total chunk time: {chunk_time:.2f}s\n")
            
            sr = sample_rate
            all_audio.append(wavs[0])
            
            # Add 400ms silence for natural rhythm
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
