import os

import runpod
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import base64
import json
import io
import time
import re
import soundfile as sf
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding

# ==========================================
# CONFIGURATION
# ==========================================
# Model and Voice Paths
VOICE_ROOT = Path("/runpod-volume/user_voices")
PRESET_ROOT = Path("/runpod-volume/preset_voices")

# --- SMART PATH DISCOVERY ---
# Use existing 1.7B Base model
MODEL_NAME_OR_PATH = "/runpod-volume/qwen3_models"  # Your existing 1.7B

if not (Path(MODEL_NAME_OR_PATH) / "model.safetensors").exists():
    MODEL_NAME_OR_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # Fallback to repo
    print(f"[startup] Model not found locally, will download: {MODEL_NAME_OR_PATH}")
else:
    print(f"[startup] Using existing 1.7B Base model: {MODEL_NAME_OR_PATH}")

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
    global model, torch
    if model is not None:
        return
        
    try:
        from qwen_tts import Qwen3TTSModel
        
        # Persistence Optimization: Cache compiled kernels on the network volume
        # This means only ONE worker EVER needs to compile. All others will load from disk.
        cache_dir = "/runpod-volume/.torch_compile_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        
        print(f"[startup] Loading Qwen3-TTS model: {MODEL_NAME_OR_PATH}...")
        print(f"[startup] Persistent cache enabled at: {cache_dir}")
        
        # Speed Optimization: Enable TF32 for faster math on modern GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Silence the 'Scary' Dynamo warnings early (they are normal for complex models)
        try:
            import logging
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
        except:
            pass
        
        # Use PyTorch native SDPA (Scaled Dot Product Attention)
        # No flash-attn compilation needed — works on T4, A10G, L4, etc.
        model = Qwen3TTSModel.from_pretrained(
            str(MODEL_NAME_OR_PATH),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )

        # Reduce VRAM pressure for serverless concurrency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"[VRAM] Clean slate: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
        
        # Configure tokenizer
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            model.tokenizer.padding_side = 'left'
            model.tokenizer.pad_token = model.tokenizer.eos_token

        # --- ATTENTION VERIFICATION ---
        print(f"\n{'='*60}")
        print(f"[SDPA] 🔍 Verifying Attention Backend")
        print(f"{'='*60}")
        print(f"[SDPA] ✅ Using PyTorch native SDPA (no flash-attn needed)")
        print(f"[SDPA] Model type: {type(model).__name__}")
        if torch.cuda.is_available():
            print(f"[SDPA] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[SDPA] VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
        print(f"{'='*60}\n")

        # --- WARM-UP & COMPILATION ---
        # This forces the 'code_predictor' and 'speaker_encoder' to initialize NOW
        print("[startup] Warming up model engine...")
        dummy_ref = str(PRESET_ROOT / "/runpod-volume/preset_voices/en/Owen.wav") # Owen is reliably 24khz
        if not os.path.exists(dummy_ref):
            # Fallback to any file in presets
            for f in PRESET_ROOT.rglob("*.wav"):
                dummy_ref = str(f)
                break

        if os.path.exists(dummy_ref):
            try:
                # Initial load test (without compilation yet)
                print("[startup] Testing model base logic...")
                with torch.inference_mode():
                    model.generate_voice_clone(
                        text="Hi.", 
                        language="english", 
                        ref_audio=dummy_ref,
                        x_vector_only_mode=True, 
                        max_new_tokens=1
                    )
                print("[startup] Base logic OK.")
                
                # --- TURBO MODE: TORCH COMPILE ---
                # Check for C++ compiler first to avoid BackendCompilerFailed on some RunPod images
                import shutil
                has_compiler = shutil.which("g++") or shutil.which("clang++") or os.environ.get("CC")
                
                if has_compiler:
                    print("[startup] 🚀 Turbo Mode: Compiling generation function for speed...")
                    try:
                        # Silence Dynamo warnings
                        try:
                            import torch._logging
                            torch._logging.set_logs(dynamo=torch._logging.log_levels.ERROR)
                        except: pass

                        # CHANGE: Use mode="default". 
                        # "reduce-overhead" takes 15+ mins to compile because of CUDA Graphs.
                        # "default" is much faster (1-2 mins) and still provides a good boost.
                        model.generate_voice_clone = torch.compile(
                            model.generate_voice_clone, 
                            mode="default",
                            fullgraph=False
                        )
                        
                        # FORCE COMPILATION NOW (while starting up)
                        # We do a tiny generation to 'bake' the first kernels
                        print("[startup] ⏳ Baking optimized kernels (this prevents the first request from timing out)...")
                        with torch.inference_mode():
                            model.generate_voice_clone(
                                text="Hi.", language="english", ref_audio=dummy_ref,
                                x_vector_only_mode=True, max_new_tokens=5
                            )
                        print("[startup] ✅ Compilation & Warm-up complete!")
                    except Exception as compile_err:
                        print(f"[startup] ⚠️ Turbo Mode failed to initialize: {compile_err}")
                else:
                    print("[startup] ⚠️ Turbo Mode DISABLED: No C++ compiler (g++) found.")
                    print("[startup] Tip: Run 'apt-get install g++' for 2x faster performance.")
            except Exception as e:
                print(f"[startup] Warm-up/Compilation skipped: {e}")
        
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
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def decrypt_voice(encrypted_base64, key_base64):
    """Decrypt the voice file in memory."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_base64)
        key = base64.b64decode(key_base64)

        # First 16 bytes are the IV
        iv = encrypted_bytes[:16]
        encrypted_data = encrypted_bytes[16:]

        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

        # Remove PKCS7 padding
        unpadder = padding.PKCS7(128).unpadder()
        voice_bytes = unpadder.update(decrypted_padded) + unpadder.finalize()

        return voice_bytes
    except Exception as e:
        print(f"Decryption error: {e}")
        return None

# ==========================================
# TEXT PREPROCESSING FOR NATURAL SPEECH
# ==========================================
def preprocess_text_for_natural_speech(text):
    """Preprocess text to produce more natural-sounding, well-paced speech.
    
    This normalizes punctuation, ensures proper spacing, and adds subtle
    textual cues that guide the TTS model toward natural pauses and pacing.
    """
    import re
    
    # 1. Normalize Unicode quotes and dashes to ASCII
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # smart double quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # smart single quotes
    text = text.replace('\u2014', ' - ')   # em-dash → spaced hyphen (natural pause)
    text = text.replace('\u2013', ' - ')   # en-dash → spaced hyphen
    text = text.replace('\u2026', '...')    # ellipsis character → three dots
    
    # 2. Ensure there is always a space after sentence-ending punctuation
    #    so the model perceives clear sentence boundaries
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # 3. Normalize multiple spaces / newlines into clean boundaries
    #    - Preserve paragraph breaks (double newline) as a marker
    text = re.sub(r'\n\s*\n', '\n\n', text)   # normalize paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # single newline → space
    
    # 4. Add a gentle pause cue ("...") after paragraph breaks.
    #    The model interprets "..." as a natural hesitation/pause.
    text = text.replace('\n\n', '... ')
    
    # 5. Collapse excessive whitespace
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def normalize_audio_loudness(audio, sr, target_dbfs=-14.0):
    """Normalize audio loudness using RMS-based gain.
    
    Applies gain to bring the audio to a consistent loudness level.
    Target is -14 dBFS which is the standard for spoken word / audiobooks.
    """
    if len(audio) == 0:
        return audio
    
    # Calculate current RMS
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-8:  # Near-silent audio
        print(f"[normalize] ⚠️  Audio is near-silent (RMS={rms:.6f}), skipping normalization")
        return audio
    
    # Calculate target RMS from dBFS
    target_rms = 10 ** (target_dbfs / 20.0)  # -14 dBFS ≈ 0.1995
    
    # Calculate required gain
    gain = target_rms / rms
    
    # Cap gain to avoid amplifying noise excessively (Increased to 15x for super quiet recordings)
    max_gain = 15.0
    if gain > max_gain:
        print(f"[normalize] ⚠️  Gain capped at {max_gain}x (original would be {gain:.1f}x)")
        gain = max_gain
    
    print(f"[normalize] 🔊 RMS: {rms:.4f} → target {target_rms:.4f} (gain={gain:.2f}x)")
    
    # Apply gain
    normalized = audio * gain
    
    # Hard clip to prevent distortion
    normalized = np.clip(normalized, -0.999, 0.999).astype(np.float32)
    
    final_rms = float(np.sqrt(np.mean(normalized ** 2)))
    print(f"[normalize] ✅ Final RMS: {final_rms:.4f} ({20 * np.log10(final_rms + 1e-10):.1f} dBFS)")
    
    return normalized


def normalize_reference_audio(voice_path, sr=24000):
    """Normalize the reference voice recording for better voice DNA extraction.
    
    A low-volume reference leads to low-volume output. This ensures the
    voice signature is extracted from a well-leveled signal.
    """
    import torchaudio
    try:
        waveform, sample_rate_ref = torchaudio.load(str(voice_path))
        audio_np = waveform.squeeze().numpy()
        
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        if rms < 0.15:  # Higher threshold to ensure robust reference
            print(f"[ref_norm] 🔇 Reference audio is quiet (RMS={rms:.4f}), normalizing...")
            target_rms = 0.20  # Stronger level for reference to encourage loud output
            gain = min(target_rms / max(rms, 1e-8), 12.0) # More headroom for quiet clips
            audio_np = np.clip(audio_np * gain, -0.999, 0.999).astype(np.float32)
            
            # Save normalized reference
            import torch
            normalized_waveform = torch.from_numpy(audio_np).unsqueeze(0)
            temp_path = "/tmp/normalized_ref.wav"
            torchaudio.save(temp_path, normalized_waveform, sample_rate_ref)
            print(f"[ref_norm] ✅ Reference normalized (gain={gain:.2f}x)")
            return temp_path
        else:
            print(f"[ref_norm] ✅ Reference audio level OK (RMS={rms:.4f})")
            return str(voice_path)
    except Exception as e:
        print(f"[ref_norm] ⚠️  Reference normalization skipped: {e}")
        return str(voice_path)


def split_into_natural_chunks(text, max_chars=300):
    """Split text into chunks that respect natural speech boundaries.
    
    Priority order:
      1. Paragraph breaks ("... " markers from preprocessing)
      2. Sentence boundaries (after . ! ?)
      3. Clause boundaries (after , ; :) — only if sentence is very long
    
    Each chunk ends at a natural pause point so inter-chunk silence
    sounds like a real breath or thought break.
    """
    import re
    
    # First, split by paragraph-break markers
    paragraphs = [p.strip() for p in text.split('... ') if p.strip()]
    
    chunks = []
    for para in paragraphs:
        # Split paragraph into sentences (keep the punctuation attached)
        sentences = re.split(r'(?<=[.!?])\s+', para)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        for sent in sentences:
            # If adding this sentence would exceed the limit, flush
            if current_chunk and (len(current_chunk) + 1 + len(sent)) > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = sent
            else:
                current_chunk = (current_chunk + " " + sent).strip() if current_chunk else sent
        
        if current_chunk:
            # Mark the last chunk of each paragraph so we know to add
            # a longer pause after it
            chunks.append(current_chunk.strip())
    
    # Safety: break any remaining oversized chunks at clause boundaries
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Split at clause boundaries (comma, semicolon, colon)
            parts = re.split(r'(?<=[,;:])\s+', chunk)
            sub = ""
            for part in parts:
                if sub and (len(sub) + 1 + len(part)) > max_chars:
                    final_chunks.append(sub.strip())
                    sub = part
                else:
                    sub = (sub + " " + part).strip() if sub else part
            if sub:
                final_chunks.append(sub.strip())
    
    # CRITICAL: Merge tiny chunks into previous chunk.
    # Chunks under ~60 chars (e.g. "Gute Nacht.") sound robotic because
    # the model has too little context for natural prosody.
    MIN_CHUNK_CHARS = 60
    merged = []
    for chunk in final_chunks:
        if merged and len(chunk) < MIN_CHUNK_CHARS:
            # Merge with previous chunk
            merged[-1] = merged[-1] + " " + chunk
            print(f"[chunker] 🔗 Merged short chunk ({len(chunk)} chars) into previous")
        else:
            merged.append(chunk)
    
    return merged


def get_pause_duration_after_chunk(chunk_text, sr):
    """Return silence samples to insert after a chunk, based on how it ends.
    
    Longer pauses after paragraph-like endings, medium after sentences,
    shorter after clause boundaries (comma, semicolon).
    """
    text = chunk_text.rstrip()
    
    if text.endswith('...'):
        # Paragraph transition / ellipsis — longest pause
        return np.zeros(int(0.65 * sr))
    elif text.endswith(('.', '!', '?')):
        # Sentence end — comfortable breath pause
        return np.zeros(int(0.45 * sr))
    elif text.endswith((',', ';', ':', '-')):
        # Clause boundary — short, natural pause
        return np.zeros(int(0.25 * sr))
    else:
        # Default — moderate pause
        return np.zeros(int(0.35 * sr))


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
    
    # Encryption parameters
    encrypted_voice = inp.get("encrypted_voice")
    encryption_key = inp.get("encryption_key")
    
    # Translate language code to full name (e.g. 'de' -> 'german')
    language = LANG_MAP.get(lang_code.lower(), "auto")
    
    # --- PROMPT LEAK PROTECTION ---
    if text and len(text) > 100 and (("The user wants" in text and "child" in text) or ("Story Details:" in text)):
        print(f"\n{'!'*60}")
        print(f"[TTS] ⚠️  PROMPT LEAK DETECTED!")
        print(f"[TTS] The input text looks like an internal LLM prompt, not a story.")
        print(f"[TTS] Leak preview: {text[:120]}...")
        print(f"{'!'*60}\n")
    
    print(f"[TTS] Initializing generation for: {text[:50]}...")
    
    # --- VOICE SETTINGS ---
    # Tuned for expressive, engaging storytelling:
    #   - Temperature 0.75 → adds just enough "acting" and pitch variety
    #   - Top_p 0.95 → allows for a richer range of natural inflections
    #   - Repetition_penalty 1.05 → keeps it stable without being robotic
    temperature = float(inp.get("temperature", 0.75)) 
    top_p = float(inp.get("top_p", 0.95))             
    repetition_penalty = float(inp.get("repetition_penalty", 1.05)) 
    top_k = int(inp.get("top_k", 50))
    
    # Subtalker (acoustic code predictor) settings — tuned for energy!
    subtalker_temperature = float(inp.get("subtalker_temperature", 0.75))
    subtalker_top_k = int(inp.get("subtalker_top_k", 50))
    subtalker_top_p = float(inp.get("subtalker_top_p", 0.95))
    subtalker_repetition_penalty = float(inp.get("subtalker_repetition_penalty", 1.05))
    
    if not text:
        return {"error": "No text provided"}

        return {"error": "No text provided"}

    # Find prompt audio
    voice_path = None
    
    # Case 1: Encrypted Voice (New Secure Flow)
    if encrypted_voice and encryption_key:
        print(f"[TTS] 🔒 Decrypting secure voice in memory...")
        voice_bytes = decrypt_voice(encrypted_voice, encryption_key)
        if not voice_bytes:
            return {"error": "Failed to decrypt voice"}
            
        # Create a temp file in RAM disk (/dev/shm) to avoid touching physical disk
        # Qwen3TTSModel likely needs a file path
        # Using a unique name to avoid collisions in concurrent requests
        temp_voice_path = f"/dev/shm/temp_voice_{int(time.time()*1000)}_{os.getpid()}.wav"
        with open(temp_voice_path, "wb") as f:
            f.write(voice_bytes)
        voice_path = Path(temp_voice_path)
        print(f"[TTS] ✅ Voice decrypted to RAM disk: {voice_path}")
        
    # Case 2: Standard Lookup (Legacy/Preset)
    else:
        voice_path = find_voice(user_id=user_id, voice_id=voice_id, preset=preset_voice)
        
    if not voice_path:
        return {"error": f"Voice not found: {preset_voice or voice_id}"}

    # --- NATURAL SPEECH PREPROCESSING ---
    # Preprocess text for better prosody and natural pauses
    text = preprocess_text_for_natural_speech(text)
    print(f"[TTS] 📝 Preprocessed text ({len(text)} chars): '{text[:100]}...'")
    
    # --- THE STITCHER: SMART NATURAL CHUNKING ---
    text_chunks = split_into_natural_chunks(text, max_chars=280)
    
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

        # --- VOICE DNA CACHING (create_voice_clone_prompt) ---
        # Pre-encode voice signature ONCE and reuse across all chunks.
        # This ensures consistent voice across chunks AND is faster.
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

        # Normalize reference audio volume (quiet recordings → quiet output)
        voice_path = normalize_reference_audio(voice_path)

        # Build a reusable voice clone prompt (voice DNA cache)
        voice_clone_prompt = None
        try:
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=str(voice_path),
                ref_text=None,
                x_vector_only_mode=True,
            )
            prep_time = time.time() - prep_start
            print(f"[TTS] ✅ Voice DNA cached in {prep_time:.2f}s (reused for all {len(text_chunks)} chunks)")
        except Exception as e:
            print(f"[TTS] ⚠️  Voice prompt caching failed, falling back to per-chunk encoding: {e}")

        # Ensure we use high-speed no-grad mode
        torch.set_grad_enabled(False)

        # Add a small leading silence for a natural start (150ms)
        lead_silence = np.zeros(int(0.15 * sr))
        all_audio.append(lead_silence)

        print(f"\n{'='*60}")
        print(f"[PROFILER] 🔍 DEEP PERFORMANCE ANALYSIS")
        print(f"{'='*60}\n")

        for i, chunk in enumerate(text_chunks):
            # Log progress for UI
            progress_pct = int((i / len(text_chunks)) * 90)
            log_progress("generate", progress_pct, f"Storyteller speaking (Chunk {i+1}/{len(text_chunks)})...")
            
            print(f"\n[PROFILER] --- Chunk {i+1}/{len(text_chunks)} ---")
            print(f"[PROFILER] Text: '{chunk[:80]}...' ({len(chunk)} chars)")
            
            chunk_start = time.time()
            # Token limit: slightly more generous to allow the model to breathe
            limit = int(len(chunk) * 2.5) + 80
            print(f"[PROFILER] Token limit: {limit}")
            
            # Sync GPU before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            t0 = time.time()
            print(f"[PROFILER] ⏱️  Starting generation at t={0:.2f}s...")
            
            # Generate this specific chunk
            # Use cached voice prompt if available, otherwise fall back
            gen_kwargs = dict(
                text=chunk,
                language=language,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                max_new_tokens=limit,
                # Subtalker settings for consistent voice across ALL chunks
                subtalker_dosample=True,
                subtalker_temperature=subtalker_temperature,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_repetition_penalty=subtalker_repetition_penalty,
            )
            
            if voice_clone_prompt is not None:
                gen_kwargs["voice_clone_prompt"] = voice_clone_prompt
            else:
                gen_kwargs["ref_audio"] = str(voice_path)
                gen_kwargs["ref_text"] = None
                gen_kwargs["x_vector_only_mode"] = True
            
            # Generate this specific chunk with Turbo Speed
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                wavs, sample_rate = model.generate_voice_clone(**gen_kwargs)
            
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
            
            # Add variable silence based on how the chunk ends
            # (paragraph break > sentence end > clause boundary)
            if i < len(text_chunks) - 1:  # No trailing silence after last chunk
                pause = get_pause_duration_after_chunk(chunk, sr)
                all_audio.append(pause)
                pause_ms = int(len(pause) / sr * 1000)
                print(f"[TTS] 🔇 Added {pause_ms}ms pause after chunk {i+1}")
            
        # Combine pieces
        final_audio = np.concatenate(all_audio)
        
        # --- LOUDNESS NORMALIZATION ---
        # Normalize the final audio to a consistent volume level (-14 dBFS)
        # This fixes the "audio is too quiet" issue
        print(f"[TTS] 🔊 Normalizing final audio loudness...")
        final_audio = normalize_audio_loudness(final_audio, sr, target_dbfs=-14.0)
        
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
    finally:
        # Cleanup temp file if it was created in /dev/shm
        if voice_path and str(voice_path).startswith("/dev/shm/") and os.path.exists(voice_path):
            try:
                os.remove(voice_path)
                print(f"[Cleanup] Removed temp voice file: {voice_path}")
            except Exception as e:
                print(f"[Cleanup] Error removing temp file: {e}")

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
