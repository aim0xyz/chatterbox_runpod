import os
import uuid
import base64
import io
import re
import random
import subprocess
import sys
import traceback
import json
from pathlib import Path

import runpod
import torch
import soundfile as sf
import numpy as np

print(f"[startup] Python: {sys.version}")

# --- Import Multilingual TTS ---
Chatterbox = None
MODEL_TYPE = "unknown"

try:
    from chatterbox.tts import ChatterboxTTS as Chatterbox
    MODEL_TYPE = "english"
    print("[startup] ChatterboxTTS (English) imported")
except Exception as e:
    print(f"[warn] English TTS import failed: {e}")

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxMTL
    MODEL_TYPE = "multilingual"
    print("[startup] ChatterboxMultilingualTTS imported - will use this")
except Exception as e:
    ChatterboxMTL = None
    print(f"[warn] Multilingual TTS import failed: {e}")

MODEL_PATH = Path("/runpod-volume/models")
VOICE_ROOT = Path("/runpod-volume/user_voices")
PRESET_ROOT = Path("/runpod-volume/preset_voices")
SAMPLE_RATE = 24000

VOICE_ROOT.mkdir(parents=True, exist_ok=True)
PRESET_ROOT.mkdir(parents=True, exist_ok=True)

print(f"[config] Models: {MODEL_PATH}")
print(f"[config] User voices: {VOICE_ROOT}")
print(f"[config] Presets: {PRESET_ROOT}")
print(f"[config] Model type: {MODEL_TYPE}")

model_english = None
model_multilingual = None

def load_model(language="en"):
    global model_english, model_multilingual
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Enable CUDA optimizations for faster inference
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        print(f"[model] CUDA optimizations enabled")
    
    # Use multilingual model for non-English languages
    if language != "en" and ChatterboxMTL is not None:
        if model_multilingual is None:
            print(f"[model] Loading MULTILINGUAL model on {device}...")
            model_multilingual = ChatterboxMTL.from_pretrained(device)
            # Compile model for faster inference (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile') and device == "cuda":
                    print("[model] Compiling model for faster inference...")
                    model_multilingual = torch.compile(model_multilingual, mode="reduce-overhead")
                    print("[model] Model compiled successfully")
            except Exception as e:
                print(f"[model] Model compilation skipped: {e}")
            print("[model] Multilingual model loaded")
        return model_multilingual, "multilingual"
    
    # Use English model
    if model_english is None:
        if Chatterbox is None:
            raise RuntimeError("No TTS model available")
        print(f"[model] Loading ENGLISH model on {device}...")
        model_english = Chatterbox.from_pretrained(device)
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and device == "cuda":
                print("[model] Compiling model for faster inference...")
                model_english = torch.compile(model_english, mode="reduce-overhead")
                print("[model] Model compiled successfully")
        except Exception as e:
            print(f"[model] Model compilation skipped: {e}")
        print("[model] English model loaded")
    return model_english, "english"

def get_user_dir(user_id):
    safe = "".join(c for c in user_id if c.isalnum() or c in "_-")
    d = VOICE_ROOT / safe
    d.mkdir(parents=True, exist_ok=True)
    return d

def find_voice(user_id=None, filename=None, preset=None):
    if preset:
        for ext in ['', '.wav', '.mp3', '.flac']:
            p = PRESET_ROOT / f"{preset}{ext}"
            if p.exists():
                return p
        for f in PRESET_ROOT.rglob(f"*{preset}*"):
            if f.is_file():
                return f
    if user_id and filename:
        p = get_user_dir(user_id) / filename
        if p.exists():
            return p
    return None

def get_voice_language(user_id=None, filename=None):
    """Get the stored language for a voice file from metadata."""
    if not user_id or not filename:
        return None
    
    try:
        user_dir = get_user_dir(user_id)
        # Look for metadata file: voice_name.json
        voice_name = Path(filename).stem  # Remove extension
        metadata_path = user_dir / f"{voice_name}.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get("language")
    except Exception as e:
        print(f"[voice_lang] Error reading voice metadata: {e}")
    
    return None

def save_voice_language(user_id, filename, language):
    """Save the language metadata for a voice file."""
    try:
        user_dir = get_user_dir(user_id)
        voice_name = Path(filename).stem  # Remove extension
        metadata_path = user_dir / f"{voice_name}.json"
        
        metadata = {"language": language, "filename": filename}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"[voice_lang] Saved language metadata: {language} for {filename}")
    except Exception as e:
        print(f"[voice_lang] Error saving voice metadata: {e}")

def clean_text(text):
    replacements = {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u2014": " - ", "\u2013": "-",
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r'\s+', ' ', text).strip()

def split_sentences(text):
    if not text:
        return []
    text = text.strip()
    if text and text[-1] not in '.!?':
        text += '.'
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_sentences(sentences, max_chars=180):
    if not sentences:
        return []
    chunks = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
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

def apply_fade(wav, fade_ms=50):
    """Apply smoother fade in/out to reduce clicks and pops."""
    if len(wav) < 200:
        return wav
    wav = wav.copy()
    fade_samples = min(int(SAMPLE_RATE * fade_ms / 1000), len(wav) // 4)
    if fade_samples > 0:
        # Use cosine curve for smoother fade (ease in/out)
        fade_in = np.cos(np.linspace(np.pi, 0, fade_samples))
        fade_in = (1 - fade_in) / 2  # Normalize to 0-1
        fade_out = np.cos(np.linspace(0, np.pi, fade_samples))
        fade_out = (1 - fade_out) / 2  # Normalize to 0-1
        
        wav[:fade_samples] *= fade_in.astype(np.float32)
        wav[-fade_samples:] *= fade_out.astype(np.float32)
    return wav

def normalize_chunk(wav, target_rms=0.12):
    """Normalize a single chunk to target RMS level for consistency."""
    if len(wav) == 0:
        return wav
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms > 1e-6:
        gain = min(2.0, max(0.5, target_rms / rms))  # Clamp gain to reasonable range
        wav = (wav * gain).astype(np.float32)
        # Prevent clipping
        max_amp = np.max(np.abs(wav))
        if max_amp > 0.95:
            wav = wav * (0.95 / max_amp)
    return wav

def apply_high_pass_filter(wav, cutoff_hz=80):
    """Apply a gentle high-pass filter to remove low-frequency artifacts (optimized)."""
    if len(wav) < 100:
        return wav
    
    # Simple first-order high-pass filter (IIR) - vectorized for speed
    # This removes DC offset and very low-frequency rumble
    alpha = 1.0 / (1.0 + 2 * np.pi * cutoff_hz / SAMPLE_RATE)
    filtered = np.zeros_like(wav, dtype=np.float32)
    filtered[0] = wav[0]
    
    # Vectorized computation for better performance
    diff = np.diff(wav, prepend=wav[0])
    for i in range(1, len(wav)):
        filtered[i] = alpha * (filtered[i-1] + diff[i])
    
    return filtered

def crossfade(audio1, audio2, crossfade_ms=50):
    """Smoothly crossfade between two audio segments with longer overlap."""
    crossfade_samples = int(SAMPLE_RATE * crossfade_ms / 1000)
    if len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        return np.concatenate([audio1, audio2])
    
    # Use longer crossfade with smoother curves (raised cosine)
    # This creates a more natural-sounding transition
    t = np.linspace(0, 1, crossfade_samples)
    fade_out = 0.5 * (1 + np.cos(np.pi * t))  # Raised cosine fade out
    fade_in = 0.5 * (1 - np.cos(np.pi * t))    # Raised cosine fade in
    
    # Apply crossfade
    audio1_end = audio1[-crossfade_samples:] * fade_out
    audio2_start = audio2[:crossfade_samples] * fade_in
    
    # Combine with energy normalization to prevent volume jumps
    combined = audio1_end + audio2_start
    # Normalize to prevent clipping
    max_amp = np.max(np.abs(combined))
    if max_amp > 0.95:
        combined = combined * (0.95 / max_amp)
    
    # Concatenate: audio1 (without end) + crossfaded + audio2 (without start)
    return np.concatenate([
        audio1[:-crossfade_samples],
        combined,
        audio2[crossfade_samples:]
    ]).astype(np.float32)

def stitch_chunks(audio_list, pause_ms=100):
    """Stitch audio chunks with seamless crossfading and minimal pauses."""
    if not audio_list:
        return np.array([], dtype=np.float32)
    
    print(f"[stitch] Stitching {len(audio_list)} chunks with {pause_ms}ms pause")
    
    # Normalize each chunk individually for consistent volume
    normalized_chunks = []
    for i, chunk in enumerate(audio_list):
        # Apply high-pass filter first to remove low-frequency artifacts
        filtered = apply_high_pass_filter(chunk.copy())
        rms_before = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        normalized = normalize_chunk(filtered)
        rms_after = float(np.sqrt(np.mean(normalized ** 2))) if len(normalized) > 0 else 0.0
        # Longer fade for smoother edges
        faded = apply_fade(normalized, fade_ms=80)
        normalized_chunks.append(faded)
        if i < 3:  # Log first few chunks for debugging
            print(f"[stitch] Chunk {i+1}: {len(chunk)} samples, RMS {rms_before:.4f} -> {rms_after:.4f}")
    
    if len(normalized_chunks) == 1:
        return normalized_chunks[0]
    
    # Stitch with seamless crossfading and minimal pauses
    result = normalized_chunks[0]
    pause_samples = int(SAMPLE_RATE * pause_ms / 1000)
    
    import time
    stitch_start = time.time()
    
    for i, chunk in enumerate(normalized_chunks[1:], 1):
        # Short pause with crossfade for natural speech rhythm
        if pause_samples > 0:
            pause = np.zeros(pause_samples, dtype=np.float32)
            result = np.concatenate([result, pause])
        
        # Crossfade into next chunk for seamless transition
        if len(result) > 0 and len(chunk) > 0:
            result = crossfade(result, chunk, crossfade_ms=60)
        else:
            result = np.concatenate([result, chunk])
    
    stitch_time = time.time() - stitch_start
    print(f"[stitch] Stitching completed in {stitch_time:.2f}s")
    
    # Optional: Apply gentle smoothing only at transition points (disabled by default for speed)
    # The crossfading should handle most artifacts, so this is only needed if issues persist
    # Uncomment the code below if you still hear artifacts after crossfading:
    #
    # if len(result) > 200 and len(normalized_chunks) > 1:
    #     transition_window = int(SAMPLE_RATE * 0.05)  # 50ms window
    #     smoothed = result.copy()
    #     current_pos = len(normalized_chunks[0])
    #     for i in range(1, len(normalized_chunks)):
    #         start_idx = max(0, current_pos - transition_window)
    #         end_idx = min(len(smoothed), current_pos + transition_window)
    #         if end_idx > start_idx + 10:
    #             # Vectorized smoothing for this window
    #             window = 3
    #             for j in range(start_idx + window, end_idx - window):
    #                 smoothed[j] = np.mean(result[j-window:j+window+1])
    #         current_pos += len(normalized_chunks[i]) + pause_samples
    #     result = smoothed.astype(np.float32)
    
    final_rms = float(np.sqrt(np.mean(result ** 2))) if len(result) > 0 else 0.0
    final_peak = float(np.max(np.abs(result))) if len(result) > 0 else 0.0
    print(f"[stitch] Final stitched audio: {len(result)} samples, RMS: {final_rms:.4f}, Peak: {final_peak:.4f}")
    return result

def generate_tts_handler(job):
    try:
        inp = job.get("input", {})
        text = inp.get("text")
        if not text:
            return {"error": "text is required"}

        preset_voice = inp.get("preset_voice")
        user_id = inp.get("user_id")
        embedding_filename = inp.get("embedding_filename")
        
        # Language parameter - default to English
        language = inp.get("language", "en")
        
        # Voice language parameter - language of the voice sample (for accent control)
        # First try to get from input parameter, then try to auto-detect from stored metadata,
        # finally default to target language
        voice_language = inp.get("voice_language")
        
        # Auto-detect from stored metadata if not provided
        if voice_language is None and user_id and embedding_filename:
            stored_language = get_voice_language(user_id, embedding_filename)
            if stored_language:
                voice_language = stored_language
                print(f"[tts] Auto-detected voice language from metadata: {voice_language}")
        
        # Default to target language if still not found
        if voice_language is None:
            voice_language = language
            print(f"[tts] Voice language not specified, defaulting to target language: {language}")
        
        # Automatic accent control based on language matching:
        # - If voice_language == language: preserve voice accent (same language, keep natural accent)
        # - If voice_language != language: use target language accent (different language, prioritize target)
        # This can be overridden by explicitly setting preserve_voice_accent
        languages_match = voice_language == language
        
        # Preserve voice accent parameter - if True, keeps the accent from the voice sample
        # (e.g., grandma's German accent even when generating English text)
        # If False, uses target language accent (default behavior for cross-language)
        # Auto-detect: preserve accent when languages match, use target accent when they differ
        if "preserve_voice_accent" in inp:
            # Explicitly set by user
            preserve_voice_accent = bool(inp.get("preserve_voice_accent", False))
        else:
            # Auto-detect: preserve accent when languages match
            preserve_voice_accent = languages_match
        
        # Accent control parameter - controls how much to prioritize target language accent
        # vs voice sample accent (0.0 = use voice accent, 1.0 = prioritize target language accent)
        # If preserve_voice_accent is True, accent_control is set to 0.0
        # Default: 0.85 (strong accent control when different from voice) for consistent storytelling
        # But if preserve_voice_accent is True, use 0.0 to keep the voice's natural accent
        if preserve_voice_accent:
            accent_control = 0.0
            print(f"[tts] ✅ Auto-detected: Preserving voice accent (languages match: {voice_language} == {language})")
        else:
            accent_control = float(inp.get("accent_control", 0.85 if not languages_match else 0.0))
            if not languages_match:
                print(f"[tts] ⚠️  Auto-detected: Using target language accent (languages differ: {voice_language} != {language})")
        
        # Volume normalization parameter - default to True for backward compatibility
        normalize_volume = inp.get("normalize_volume", True)

        exaggeration = float(inp.get("exaggeration", 0.5))
        temperature = float(inp.get("temperature", 0.7))
        cfg_weight = float(inp.get("cfg_weight", 0.5))

        # Use character-based chunking (default ~180 chars) to keep each TTS call
        # reasonably short and avoid very long generations that can trigger the
        # model's early‑stopping heuristics.
        max_chunk_chars = int(inp.get("max_chunk_chars", 180))
        pause_ms = int(inp.get("pause_ms", 100))  # Minimal pause for seamless transitions

        print(f"[tts] Text length: {len(text)}")
        print(f"[tts] Target language: {language}")
        print(f"[tts] Voice sample language: {voice_language}")
        if languages_match:
            print(f"[tts] ✅ Languages match ({voice_language} == {language}): Preserving voice accent automatically")
        else:
            print(f"[tts] ⚠️  Languages differ ({voice_language} != {language}): Using target language accent automatically")
            print(f"[tts] Accent control: {accent_control:.2f} (will prioritize {language} accent for consistency)")

        voice_path = find_voice(user_id, embedding_filename, preset_voice)
        if not voice_path:
            return {"error": "Voice file not found"}

        print(f"[tts] Voice: {voice_path}")

        text = clean_text(text)
        sentences = split_sentences(text)
        chunks = chunk_sentences(sentences, max_chars=max_chunk_chars)

        print(f"[tts] Created {len(chunks)} chunks (max_chunk_chars={max_chunk_chars})")

        if not chunks:
            return {"error": "No text to process"}

        # Load appropriate model based on language
        tts, model_type = load_model(language)
        print(f"[tts] Using {model_type} model for language: {language}")
        
        # Calculate accent control adjustments ONCE before the loop for consistency
        # This ensures all chunks use the same parameters and produce consistent accent
        base_temperature = temperature
        base_cfg_weight = cfg_weight
        base_exaggeration = exaggeration
        
        if model_type == "multilingual" and voice_language != language:
            if preserve_voice_accent or accent_control < 0.1:
                # Preserve voice accent: use original parameters to keep voice's natural accent
                # No adjustments needed - let the voice embedding control the accent
                print(f"[tts] Preserving voice accent: using original parameters to keep {voice_language} accent")
                print(f"[tts] Parameters: temp={base_temperature:.2f}, cfg={base_cfg_weight:.2f}, exaggeration={base_exaggeration:.2f}")
            elif accent_control > 0.0:
                # More aggressive adjustments for consistent target language accent across all chunks
                # Higher temperature = more variation = less voice accent influence
                base_temperature = min(temperature + (accent_control * 0.3), 1.0)
                
                # Lower cfg_weight = less voice embedding influence = more language model control
                base_cfg_weight = max(cfg_weight * (1.0 - accent_control * 0.4), 0.15)
                
                # Slightly reduce exaggeration to make accent more consistent
                base_exaggeration = exaggeration * (1.0 - accent_control * 0.1)
                
                print(f"[tts] Accent control active: temp={base_temperature:.2f} (was {temperature:.2f}), "
                      f"cfg={base_cfg_weight:.2f} (was {cfg_weight:.2f}), "
                      f"exaggeration={base_exaggeration:.2f} (was {exaggeration:.2f})")
                print(f"[tts] These parameters will be applied consistently to ALL chunks")
        
        audio_chunks = []
        
        import time
        total_gen_start = time.time()

        for i, chunk_text in enumerate(chunks):
            chunk_start = time.time()
            print(f"[tts] Generating chunk {i+1}/{len(chunks)}: {chunk_text[:50]}...")
            try:
                # Use consistent seed per chunk for reproducibility, but ensure accent control
                # parameters are the same across all chunks
                seed = (12345 + i * 9973) & 0xFFFFFFFF
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                # Use inference_mode for faster inference (no gradient tracking)
                with torch.inference_mode():
                    # Different generate call based on model type
                    if model_type == "multilingual":
                        # Use the pre-calculated accent-controlled parameters for consistency
                        output = tts.generate(
                            text=chunk_text,
                            audio_prompt_path=str(voice_path),
                            language_id=language,
                            exaggeration=base_exaggeration,
                            temperature=base_temperature,
                            cfg_weight=base_cfg_weight,
                        )
                    else:
                        # English model doesn't use language_id
                        output = tts.generate(
                            text=chunk_text,
                            audio_prompt_path=str(voice_path),
                            exaggeration=exaggeration,
                            temperature=temperature,
                            cfg_weight=cfg_weight,
                        )

                wav = to_numpy(output)
                chunk_time = time.time() - chunk_start
                print(f"[tts]   Generated {len(wav)} samples in {chunk_time:.2f}s")

                if len(wav) > 0:
                    audio_chunks.append(wav)
            except Exception as e:
                print(f"[tts]   Error: {e}")
                traceback.print_exc()
        
        total_gen_time = time.time() - total_gen_start
        print(f"[tts] Total generation time: {total_gen_time:.2f}s ({total_gen_time/len(chunks):.2f}s per chunk)")

        if not audio_chunks:
            return {"error": "No audio generated"}

        final_wav = stitch_chunks(audio_chunks, pause_ms=pause_ms)

        # --- Volume normalization (server-side) ---
        # Normalizes audio to consistent loudness levels
        # More aggressive for user voices which tend to be quieter
        if normalize_volume and len(final_wav) > 0:
            max_amp = np.max(np.abs(final_wav))
            rms = float(np.sqrt(np.mean(final_wav ** 2)))

            # Determine if this is a user voice (more aggressive normalization needed)
            is_user_voice = user_id is not None and embedding_filename is not None
            
            # More aggressive targets for better normalization
            # User voices typically need more boost
            if is_user_voice:
                target_peak = 0.95  # Allow slightly higher peaks for user voices
                target_rms = 0.15  # Higher target RMS for user voices (~-16 dBFS)
                max_gain = 3.0      # Allow up to ~+10 dB boost for very quiet user voices
                min_gain = 0.3     # Can reduce very loud audio more
            else:
                # Preset voices - more conservative
                target_peak = 0.9
                target_rms = 0.12  # Higher than before (~-18 dBFS)
                max_gain = 2.5     # Allow more boost than before
                min_gain = 0.4

            gain = 1.0

            # If peak is too high, reduce to target_peak
            if max_amp > 1e-6 and max_amp > target_peak:
                gain = min(gain, target_peak / max_amp)

            # If overall RMS is low, boost to target_rms
            if rms > 1e-6 and rms < target_rms:
                calculated_gain = target_rms / rms
                # For very quiet audio (RMS < 0.05), apply extra boost
                if rms < 0.05:
                    calculated_gain *= 1.3  # Extra 30% boost for very quiet audio
                gain = min(max_gain, max(gain, calculated_gain))

            # Clamp to safe range
            gain = max(min_gain, min(gain, max_gain))

            if abs(gain - 1.0) > 1e-3:
                voice_type = "user voice" if is_user_voice else "preset voice"
                print(f"[tts] Applying volume normalization ({voice_type}), gain={gain:.2f}, "
                      f"rms={rms:.4f}, peak={max_amp:.4f}, target_rms={target_rms:.3f}")
                final_wav = (final_wav * gain).astype(np.float32)
                
                # Ensure we don't clip after gain application
                max_after = np.max(np.abs(final_wav))
                if max_after > 0.99:
                    # Soft limit to prevent clipping
                    final_wav = final_wav * (0.99 / max_after)
                    print(f"[tts] Applied soft limiter to prevent clipping (peak was {max_after:.3f})")
        elif not normalize_volume:
            print(f"[tts] Volume normalization disabled by request")

        buf = io.BytesIO()
        sf.write(buf, final_wav, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode('utf-8')

        duration_sec = len(final_wav) / SAMPLE_RATE
        print(f"[tts] Complete: {duration_sec:.1f}s")

        return {
            "status": "success",
            "audio": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "format": "wav",
            "duration_seconds": round(duration_sec, 2),
            "chunks_generated": len(audio_chunks),
            "chunks_requested": len(chunks),
            "language": language,
            "voice_language": voice_language,
            "preserve_voice_accent": preserve_voice_accent,
            "accent_control": accent_control,
            "model_type": model_type,
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
        # Language of the voice sample (for accent control)
        voice_language = inp.get("voice_language", "en")  # Default to English if not specified

        if not audio_b64:
            return {"error": "audio_data is required"}

        print(f"[clone] User: {user_id}, Voice: {voice_name}, Language: {voice_language}")

        user_dir = get_user_dir(user_id)
        temp_path = user_dir / f"temp_{uuid.uuid4().hex}.bin"

        with open(temp_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        final_name = f"{voice_name}_{uuid.uuid4().hex[:6]}.wav"
        final_path = user_dir / final_name

        try:
            # Convert uploaded audio to 24 kHz mono WAV for cloning
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp_path),
                "-ar", str(SAMPLE_RATE), "-ac", "1", str(final_path)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            temp_path.unlink(missing_ok=True)
            return {"error": f"FFmpeg failed: {e.stderr.decode() if e.stderr else str(e)}"}

        # We no longer need the temp container file
        temp_path.unlink(missing_ok=True)

        # --- Normalize recorded voice loudness for better cloning quality ---
        try:
            if final_path.exists():
                print(f"[clone] Normalizing recording loudness for {final_name}")
                wav, sr = sf.read(str(final_path), always_2d=False)

                # Ensure mono
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)

                # Convert to float32 in [-1, 1]
                if wav.dtype != np.float32:
                    wav = wav.astype(np.float32)

                # Compute RMS
                rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) > 0 else 0.0
                print(f"[clone]   Original RMS={rms:.4f}, samples={len(wav)}, sr={sr}")

                if rms > 0.0:
                    # Target a fairly strong but safe RMS so user voices don't sound too quiet.
                    target_rms = 0.18  # ~-15 dBFS
                    gain = target_rms / rms

                    # Only boost; don't attenuate louder recordings here
                    if gain < 1.0:
                        gain = 1.0

                    # Cap gain to avoid extreme amplification/noise
                    # Increased from 4.0 to 6.0 to handle very quiet recordings better
                    # For RMS=0.0233, we need ~7.7x to reach 0.18, so 6.0x gets us closer (~0.14 RMS)
                    max_gain = 6.0  # ~+15.6 dB (increased from +12 dB)
                    if gain > max_gain:
                        gain = max_gain

                    print(f"[clone]   Applying gain={gain:.2f} to recording")
                    wav *= gain

                    # Hard clip to safe range
                    wav = np.clip(wav, -0.999, 0.999).astype(np.float32)

                    # Save back as WAV with original or SAMPLE_RATE
                    out_sr = sr if sr and sr > 0 else SAMPLE_RATE
                    sf.write(str(final_path), wav, out_sr, format="WAV")

                    # Log final RMS
                    final_rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) > 0 else 0.0
                    print(f"[clone]   Normalized RMS={final_rms:.4f} (target={target_rms:.2f})")
                else:
                    print(f"[clone]   RMS is zero, skipping normalization for {final_name}")
        except Exception as e:
            # Don't fail the whole clone if normalization has issues
            print(f"[clone] ⚠️ Error normalizing recording loudness: {e}")
            traceback.print_exc()

        print(f"[clone] Saved: {final_name}")
        
        # Save voice language metadata for automatic detection during TTS generation
        save_voice_language(user_id, final_name, voice_language)

        return {
            "status": "success",
            "user_id": user_id,
            "embedding_filename": final_name,
            "voice_name": voice_name,
            "voice_language": voice_language,  # Return the language for client reference
        }

    except Exception as e:
        print(f"[error] Clone failed: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def list_preset_voices_handler(job):
    try:
        voices = []
        for path in PRESET_ROOT.rglob("*"):
            if path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                rel = str(path.relative_to(PRESET_ROOT))
                voices.append({
                    "filename": rel,
                    "name": path.stem.replace("_", " ").title(),
                })
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

        user_dir = get_user_dir(user_id)
        if not user_dir.exists():
            return {"status": "success", "user_voices": [], "count": 0}

        voices = []
        for f in user_dir.iterdir():
            if f.suffix.lower() in ['.wav', '.mp3']:
                voices.append({
                    "filename": f.name,
                    "name": f.stem.replace("_", " ").title(),
                })

        print(f"[user_voices] Found {len(voices)} voices")
        return {"status": "success", "user_voices": voices, "count": len(voices)}

    except Exception as e:
        print(f"[error] List user voices failed: {e}")
        return {"error": str(e)}

def delete_voice_handler(job):
    try:
        inp = job.get("input", {})
        user_id = inp.get("user_id")
        voice_id = inp.get("voice_id")  # This is the filename (e.g., "Mama_de42b3.wav" or "unique_3fda66.wav")

        if not user_id:
            return {"error": "user_id is required"}
        if not voice_id:
            return {"error": "voice_id is required"}

        print(f"[delete] User: {user_id}, Voice ID: {voice_id}")

        # First, try to find the voice using the helper function (same as TTS handler)
        voice_path = find_voice(user_id=user_id, filename=voice_id)
        
        # If not found, try with different extensions
        if not voice_path:
            user_dir = get_user_dir(user_id)
            print(f"[delete] Voice not found with exact filename, trying extensions...")
            print(f"[delete] User directory: {user_dir}")
            
            # List all files in user directory for debugging
            if user_dir.exists():
                all_files = list(user_dir.iterdir())
                print(f"[delete] Files in user directory: {[f.name for f in all_files]}")
            
            # Try with .wav extension if not present
            if not voice_id.endswith('.wav') and not voice_id.endswith('.mp3'):
                for ext in ['.wav', '.mp3']:
                    test_path = user_dir / f"{voice_id}{ext}"
                    if test_path.exists():
                        voice_path = test_path
                        print(f"[delete] Found voice with extension: {test_path.name}")
                        break
            
            # If still not found, try exact match
            if not voice_path:
                test_path = user_dir / voice_id
                if test_path.exists():
                    voice_path = test_path
                    print(f"[delete] Found voice with exact filename: {voice_id}")
        
        if not voice_path or not voice_path.exists():
            user_dir = get_user_dir(user_id)
            error_msg = f"Voice file not found: {voice_id}"
            if user_dir.exists():
                available_files = [f.name for f in user_dir.iterdir() if f.is_file()]
                error_msg += f". Available files: {available_files}"
            print(f"[delete] {error_msg}")
            return {"error": error_msg}

        # Delete the voice file
        print(f"[delete] Deleting file: {voice_path}")
        voice_path.unlink()
        print(f"[delete] ✅ Successfully deleted: {voice_path.name}")

        return {
            "status": "success",
            "message": f"Voice {voice_path.name} deleted successfully",
        }

    except Exception as e:
        print(f"[error] Delete voice failed: {e}")
        traceback.print_exc()
        return {"error": str(e)}

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
        elif action == "delete_voice":
            return delete_voice_handler(job)
        else:
            return {"error": f"Unknown action: {action}"}

    except Exception as e:
        print(f"[fatal] {e}")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("[startup] RunPod Handler ready")
    runpod.serverless.start({"handler": handler})