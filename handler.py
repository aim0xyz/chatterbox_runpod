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

def apply_high_pass_filter(wav, cutoff_hz=60):
    """Apply a high-pass filter to remove low-frequency artifacts (howling, rumble)."""
    if len(wav) < 100:
        return wav
    
    # Simple first-order high-pass filter (IIR) - vectorized for speed
    # Increased cutoff from 80Hz to 100Hz to better remove low-frequency artifacts
    # This removes DC offset and very low-frequency rumble/howing
    alpha = 1.0 / (1.0 + 2 * np.pi * cutoff_hz / SAMPLE_RATE)
    filtered = np.zeros_like(wav, dtype=np.float32)
    filtered[0] = wav[0]
    
    # Vectorized computation for better performance
    diff = np.diff(wav, prepend=wav[0])
    for i in range(1, len(wav)):
        filtered[i] = alpha * (filtered[i-1] + diff[i])
    
    return filtered

def apply_low_pass_filter(wav, cutoff_hz=10000):
    """Apply a low-pass filter to remove high-frequency artifacts (rubbing, scratching sounds)."""
    if len(wav) < 100:
        return wav
    
    # Simple first-order low-pass filter (IIR)
    # Removes high-frequency artifacts above 10kHz that can sound like rubbing/scratching
    # Speech is typically below 8kHz, so 10kHz cutoff preserves speech while removing artifacts
    alpha = 2 * np.pi * cutoff_hz / SAMPLE_RATE / (1.0 + 2 * np.pi * cutoff_hz / SAMPLE_RATE)
    filtered = np.zeros_like(wav, dtype=np.float32)
    filtered[0] = wav[0]
    
    for i in range(1, len(wav)):
        filtered[i] = alpha * wav[i] + (1.0 - alpha) * filtered[i-1]
    
    return filtered

def remove_dc_offset(wav):
    """Remove DC offset (constant bias) that can cause artifacts."""
    if len(wav) == 0:
        return wav
    # Calculate mean (DC offset) and subtract it
    dc_offset = np.mean(wav)
    if abs(dc_offset) > 1e-6:
        wav = wav - dc_offset
    return wav.astype(np.float32)

def apply_smoothing_filter(wav, window_size=5):
    """Apply a simple moving average to reduce rapid amplitude changes and clicks."""
    if len(wav) < window_size:
        return wav
    
    # Use a simple moving average to smooth out rapid changes
    # This helps reduce clicks, pops, and rapid amplitude jumps
    smoothed = np.zeros_like(wav, dtype=np.float32)
    half_window = window_size // 2
    
    # Handle edges
    for i in range(half_window):
        smoothed[i] = wav[i]
    
    # Apply moving average to middle samples
    for i in range(half_window, len(wav) - half_window):
        smoothed[i] = np.mean(wav[i - half_window:i + half_window + 1])
    
    # Handle edges
    for i in range(len(wav) - half_window, len(wav)):
        smoothed[i] = wav[i]
    
    return smoothed

def apply_spectral_gating(wav, threshold_db=-40):
    """Apply spectral gating to remove background noise and hiss."""
    if len(wav) < 1000:
        return wav
    
    # Calculate RMS in small windows to detect background noise
    window_size = int(SAMPLE_RATE * 0.05)  # 50ms windows
    num_windows = len(wav) // window_size
    
    if num_windows < 2:
        return wav
    
    # Calculate RMS for each window
    window_rms = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = wav[start:end]
        rms = float(np.sqrt(np.mean(window ** 2)))
        window_rms.append(rms)
    
    # Estimate noise floor (use lower percentile of RMS values)
    noise_floor = np.percentile(window_rms, 25)  # 25th percentile as noise floor
    
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold_db / 20.0)
    gate_threshold = max(noise_floor * 2, threshold_linear)  # At least 2x noise floor
    
    # Apply gating: reduce amplitude of quiet sections
    gated = wav.copy().astype(np.float32)
    for i in range(num_windows):
        start = i * window_size
        end = min(start + window_size, len(wav))
        window_rms_val = window_rms[i]
        
        if window_rms_val < gate_threshold:
            # Reduce amplitude of quiet sections (gate)
            reduction = window_rms_val / gate_threshold
            # Smooth reduction to avoid clicks
            gated[start:end] *= (0.1 + 0.9 * reduction)  # Keep at least 10% to avoid silence
    
    return gated

def detect_and_remove_clicks(wav, threshold=0.3):
    """Detect and remove sudden amplitude spikes (clicks/pops)."""
    if len(wav) < 10:
        return wav
    
    # Calculate derivative to detect sudden changes
    diff = np.abs(np.diff(wav, prepend=wav[0]))
    
    # Find samples with sudden amplitude changes
    click_threshold = threshold * np.std(diff) + np.mean(diff)
    click_indices = np.where(diff > click_threshold)[0]
    
    if len(click_indices) == 0:
        return wav
    
    # Smooth out clicks by interpolating with neighbors
    cleaned = wav.copy().astype(np.float32)
    for idx in click_indices:
        if 1 < idx < len(wav) - 1:
            # Replace with average of neighbors
            cleaned[idx] = (wav[idx - 1] + wav[idx + 1]) / 2.0
    
    return cleaned

def apply_high_shelf_filter(wav, cutoff_hz=10000, gain_db=-2):
    """
    Apply high-shelf filter to gently reduce high-frequency noise
    without making audio muffled. More subtle than a low-pass filter.
    Reduces scratchy/sandy artifacts while preserving speech clarity.
    """
    if len(wav) < 100:
        return wav
    
    fc = cutoff_hz / SAMPLE_RATE
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Calculate high-frequency component via differentiation
    diff = np.diff(wav, prepend=wav[0])
    
    # Apply gentle gain reduction to high frequencies
    # This reduces high-frequency noise while preserving speech
    diff_scaled = diff * (1.0 - (1.0 - gain_linear) * fc * 2)
    
    # Reconstruct signal
    filtered = np.cumsum(diff_scaled)
    filtered = filtered - np.mean(filtered) + np.mean(wav)
    
    return filtered.astype(np.float32)

def soft_limiter(wav, threshold=0.88, ratio=4.0):
    """
    Apply soft limiting to prevent harsh clipping artifacts.
    Compresses peaks gradually instead of hard clipping.
    Makes loud sections smoother and more natural.
    """
    if len(wav) == 0:
        return wav
    
    result = wav.copy()
    mask = np.abs(result) > threshold
    
    if np.any(mask):
        # Soft knee compression formula
        excess = np.abs(result[mask]) - threshold
        compressed = threshold + excess / ratio
        
        # Preserve sign
        result[mask] = np.sign(result[mask]) * compressed
    
    return result.astype(np.float32)

def reduce_breathing_artifacts(wav, breath_threshold_db=-35, reduction_factor=0.2):
    """
    Enhanced detection and reduction of harsh breathing sounds (inhale/exhale artifacts).
    Uses Zero-Crossing Rate (ZCR) and spectral energy balance for better accuracy.
    """
    if len(wav) < 1000:
        return wav
    
    # Window size for breath detection (typical breath is 100-300ms)
    window_size = int(SAMPLE_RATE * 0.10)  # 100ms windows
    hop_size = window_size // 2
    
    num_windows = (len(wav) - window_size) // hop_size
    if num_windows < 1:
        return wav
    
    result = wav.copy()
    threshold_linear = 10 ** (breath_threshold_db / 20.0)
    
    for i in range(num_windows):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        
        window = wav[start_idx:end_idx]
        rms = float(np.sqrt(np.mean(window ** 2)))
        
        # Breaths are quiet (low RMS) but have high zero-crossing rate (hiss-like)
        if rms < threshold_linear and rms > 1e-6:
            # Calculate Zero-Crossing Rate (ZCR)
            zcr = np.mean(np.abs(np.diff(np.sign(window)))) / 2
            
            # Breaths typically have ZCR between 0.05 and 0.4
            # Speech usually has lower ZCR (vowels) or much higher/different patterns
            if zcr > 0.08:
                # Calculate high-frequency energy ratio
                diff = np.abs(np.diff(window))
                hf_energy = float(np.mean(diff))
                
                # If high-frequency energy is dominant, it's a breath or hiss
                if hf_energy > rms * 0.6:
                    # Apply smooth reduction envelope
                    reduction_envelope = np.linspace(1.0, reduction_factor, window_size // 4)
                    middle = np.ones(window_size - (window_size // 2)) * reduction_factor
                    reduction_envelope = np.concatenate([
                        reduction_envelope,
                        middle,
                        np.linspace(reduction_factor, 1.0, window_size // 4)
                    ])
                    
                    # Ensure same length
                    if len(reduction_envelope) > window_size:
                        reduction_envelope = reduction_envelope[:window_size]
                    elif len(reduction_envelope) < window_size:
                        reduction_envelope = np.pad(reduction_envelope, (0, window_size - len(reduction_envelope)), 'edge')
                    
                    result[start_idx:end_idx] *= reduction_envelope
    
    return result.astype(np.float32)

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

def stitch_chunks(audio_list, chunk_texts, pause_ms=100):
    """Stitch audio chunks with storytelling pauses and seamless crossfading."""
    if not audio_list:
        return np.array([], dtype=np.float32)
    
    # Process and normalize each chunk individually for consistent volume
    processed_chunks = []
    for i, chunk in enumerate(audio_list):
        # Step A: High-pass filter to remove low-frequency rumbles first
        # Doing this before trimming helps the RMS check ignore sub-bass noise
        filtered = apply_high_pass_filter(chunk.copy(), cutoff_hz=100)
        
        # Step B: Energy-based Tail Trimmer (CRITICAL)
        # We trim BEFORE normalization so thumps don't squash the speech volume
        tail_threshold = 10 ** (-42 / 20.0)
        win_len = int(SAMPLE_RATE * 0.04) # 40ms window
        
        # Default boundaries
        last_speech_idx = len(filtered)
        first_speech_idx = 0
        
        # Scan backwards from the end (up to 400ms) to cut out thumps/hallucinations
        max_scan_back = int(SAMPLE_RATE * 0.4)
        for j in range(len(filtered) - win_len, max(0, len(filtered) - max_scan_back), -win_len//2):
            win_rms = np.sqrt(np.mean(filtered[j:j+win_len]**2))
            if win_rms > tail_threshold:
                last_speech_idx = j + win_len
                break
            last_speech_idx = j
            
        # Also clean up the start for inhale/click artifacts
        for j in range(0, min(len(filtered), int(SAMPLE_RATE * 0.15)), win_len//2):
            win_rms = np.sqrt(np.mean(filtered[j:j+win_len]**2))
            if win_rms > tail_threshold:
                first_speech_idx = max(0, j - win_len)
                break
            first_speech_idx = j
            
        trimmed = filtered[first_speech_idx:last_speech_idx]
        
        # Step C: Normalize the trimmed speech
        # Now that thumps are gone, the normalization will be much more accurate
        normalized = normalize_chunk(trimmed)
        
        # Step D: Apply subtle fades for seamless stitching
        faded = apply_fade(normalized, fade_ms=80)
        processed_chunks.append(faded)
    
    if len(processed_chunks) == 1:
        return processed_chunks[0]
    
    # Stitch with dynamic storytelling pauses
    result = processed_chunks[0]
    
    import time
    stitch_start = time.time()
    
    for i, chunk in enumerate(normalized_chunks[1:], 1):
        prev_text = chunk_texts[i-1].strip()
        
        # Determine specific pause based on punctuation
        # REDUCED from previous values to minimize breathing artifact zones
        current_pause = pause_ms
        if prev_text.endswith(('.', '!', '?')):
            current_pause = 400  # Reduced from 650ms - shorter pause = less breath time
        elif prev_text.endswith((',', ';', ':')):
            current_pause = 200  # Reduced from 300ms
        elif '\n' in prev_text:
            current_pause = 550  # Reduced from 900ms
        else:
            current_pause = 100  # Reduced from 150ms
            
        pause_samples = int(SAMPLE_RATE * current_pause / 1000)
        
        if pause_samples > 0:
            pause = np.zeros(pause_samples, dtype=np.float32)
            result = np.concatenate([result, pause])
        
        # Crossfade into next chunk for seamless transition
        # LONGER crossfade to better blend breathing artifacts
        if len(result) > 0 and len(chunk) > 0:
            result = crossfade(result, chunk, crossfade_ms=80)  # Increased from 40ms to 80ms
        else:
            result = np.concatenate([result, chunk])
    
    stitch_time = time.time() - stitch_start
    print(f"[stitch] Storyteller stitching completed in {stitch_time:.2f}s")
    return result.astype(np.float32)
    
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

# ============================================
# VOICE VERIFICATION FUNCTIONS
# ============================================

def check_voice_liveness(audio_path):
    """
    Acoustic anti-spoofing check using librosa.
    Detects if audio is replayed/synthetic vs. live recording.
    """
    try:
        import librosa
        
        print(f"[anti-spoof] Analyzing audio: {audio_path}")
        
        y, sr = librosa.load(str(audio_path), sr=16000)
        
        if len(y) < 1000:
            print(f"[anti-spoof] Audio too short: {len(y)} samples")
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': 'Audio too short (minimum 1000 samples required)'}
            }
        
        # Feature extraction
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.var(mfcc))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        rms = librosa.feature.rms(y=y)
        rms_std = float(np.std(rms))
        
        print(f"[anti-spoof] Features: spectral_flat={spectral_flatness:.4f}, "
              f"zcr={zcr:.4f}, mfcc_var={mfcc_var:.2f}, "
              f"rolloff={rolloff:.1f}, rms_std={rms_std:.4f}")
        
        # Scoring
        score = 0.0
        
        if 0.15 < spectral_flatness < 0.4:
            score += 0.25
        elif spectral_flatness < 0.15:
            score += 0.15
        
        if 0.04 < zcr < 0.18:
            score += 0.20
        
        if mfcc_var > 120:
            score += 0.25
        elif mfcc_var > 60:
            score += 0.15
        
        if 1200 < rolloff < 5000:
            score += 0.15
        
        if rms_std > 0.03:
            score += 0.15
        elif rms_std > 0.015:
            score += 0.10
        
        passed = score >= 0.65
        
        print(f"[anti-spoof] Score: {score:.2f}, Passed: {passed}")
        
        return {
            'passed': passed,
            'score': float(score),
            'details': {
                'spectral_flatness': spectral_flatness,
                'zero_crossing_rate': zcr,
                'mfcc_variance': mfcc_var,
                'spectral_rolloff': rolloff,
                'rms_std': rms_std,
            }
        }
        
    except Exception as e:
        print(f"[anti-spoof] Error: {e}")
        traceback.print_exc()
        return {
            'passed': False,
            'score': 0.0,
            'details': {'error': str(e)}
        }


def verify_transcript(audio_path, original_text, language='en'):
    """
    Verify that the recorded audio matches the expected text using Speech-to-Text.
    """
    try:
        from faster_whisper import WhisperModel
        
        print(f"[stt] Transcribing audio in {language}: {audio_path}")
        
        # Load Whisper model - Force CPU to avoid cuDNN library conflicts in slim container
        # Note: Short verification clips are fast enough on CPU
        device = "cpu"
        compute_type = "int8"
        
        print(f"[stt] Initializing Whisper on {device}...")
        model = WhisperModel(
            "base",
            device=device,
            compute_type=compute_type,
            download_root="/runpod-volume/.cache/whisper"
        )
        
        # Transcribe
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
        )
        
        transcript = " ".join([segment.text for segment in segments]).strip()
        
        print(f"[stt] Transcript: {transcript[:100]}...")
        print(f"[stt] Original:   {original_text[:100]}...")
        
        match_score = calculate_text_similarity(
            transcript.lower(),
            original_text.lower()
        )
        
        # info.language is the detected language code (e.g., 'en', 'de')
        detected_language = info.language
        language_probability = info.language_probability
        
        # Define pass threshold
        passed = match_score >= 0.6
        
        print(f"[stt] Match score: {match_score:.2f}, Passed: {passed}")
        print(f"[stt] Detected language: {detected_language} ({language_probability:.2f})")
        
        return {
            'match_score': float(match_score),
            'transcript': transcript,
            'original': original_text,
            'passed': passed,
            'detected_language': detected_language,
            'language_probability': float(language_probability)
        }
        
    except Exception as e:
        print(f"[stt] Error: {e}")
        traceback.print_exc()
        return {
            'match_score': 0.0,
            'transcript': '',
            'original': original_text,
            'passed': False,
            'error': str(e)
        }


def calculate_text_similarity(str1, str2):
    """Calculate text similarity using normalized Levenshtein distance."""
    str1 = re.sub(r'[^\w\s]', '', str1).strip()
    str2 = re.sub(r'[^\w\s]', '', str2).strip()
    str1 = re.sub(r'\s+', ' ', str1)
    str2 = re.sub(r'\s+', ' ', str2)
    
    if not str1 or not str2:
        return 0.0
    
    longer = str1 if len(str1) > len(str2) else str2
    shorter = str2 if len(str1) > len(str2) else str1
    
    if len(longer) == 0:
        return 1.0
    
    edit_distance = levenshtein_distance(longer, shorter)
    similarity = (len(longer) - edit_distance) / len(longer)
    
    return max(0.0, min(1.0, similarity))


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


# ============================================
# HANDLER: Verify Recording
# ============================================

def verify_recording_handler(job):
    """Verify a voice recording before cloning."""
    try:
        inp = job.get("input", {})
        audio_b64 = inp.get("audio_data")
        original_story = inp.get("original_story")
        language = inp.get("language", "en")
        user_id = inp.get("user_id", "anonymous")
        
        if not audio_b64:
            return {"error": "audio_data is required"}
        if not original_story:
            return {"error": "original_story is required"}
        
        print(f"[verify] User: {user_id}, Language: {language}")
        print(f"[verify] Story length: {len(original_story)} chars")
        
        # Save audio temporarily
        temp_dir = Path("/tmp/verify")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"verify_{uuid.uuid4().hex}.wav"
        
        audio_bytes = base64.b64decode(audio_b64)
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Convert to proper format
        converted_path = temp_dir / f"verify_{uuid.uuid4().hex}_converted.wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp_path),
                "-ar", "24000", "-ac", "1", str(converted_path)
            ], check=True, capture_output=True, timeout=30)
        except subprocess.CalledProcessError as e:
            temp_path.unlink(missing_ok=True)
            return {"error": f"Audio conversion failed: {e.stderr.decode() if e.stderr else str(e)}"}
        
        # Anti-spoofing check
        print(f"[verify] Running anti-spoofing check...")
        liveness_result = check_voice_liveness(converted_path)
        
        # Speech-to-text verification
        print(f"[verify] Running speech-to-text verification...")
        stt_result = verify_transcript(converted_path, original_story, language)
        
        # Cleanup
        temp_path.unlink(missing_ok=True)
        converted_path.unlink(missing_ok=True)
        
        overall_passed = liveness_result['passed'] and stt_result['passed']
        
        print(f"[verify] Results: liveness={liveness_result['passed']}, "
              f"stt={stt_result['passed']}, overall={overall_passed}")
        
        return {
            "status": "success",
            "verification": {
                "passed": overall_passed,
                "liveness": {
                    "passed": liveness_result['passed'],
                    "score": liveness_result['score'],
                    "details": liveness_result['details']
                },
                "transcript": {
                    "passed": stt_result['passed'],
                    "match_score": stt_result['match_score'],
                    "transcript": stt_result.get('transcript', ''),
                    "detected_language": stt_result.get('detected_language'),
                }
            }
        }
        
    except Exception as e:
        print(f"[error] Verification failed: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ============================================
# HANDLER: Clone Voice with Verification
# ============================================

def clone_voice_verified_handler(job):
    """Clone voice with built-in verification."""
    try:
        inp = job.get("input", {})
        
        print(f"[clone_verified] Step 1: Verifying recording...")
        verification_result = verify_recording_handler(job)
        
        if "error" in verification_result:
            return verification_result
        
        if not verification_result.get("verification", {}).get("passed", False):
            print(f"[clone_verified] Verification failed, rejecting clone request")
            return {
                "error": "Verification failed",
                "verification": verification_result.get("verification"),
                "reason": "Audio did not pass anti-spoofing or transcript verification"
            }
        
        print(f"[clone_verified] ✅ Verification passed, proceeding with clone...")
        
        # Use Whisper's detected language for auto accent selection
        detected_lang = verification_result.get("verification", {}).get("transcript", {}).get("detected_language")
        if detected_lang:
            print(f"[clone_verified] Overriding voice_language with detected language: {detected_lang}")
            inp["voice_language"] = detected_lang
        
        return clone_voice_handler(job)
        
    except Exception as e:
        print(f"[error] Clone with verification failed: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ============================================
# EXISTING HANDLERS
# ============================================

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
        # CRITICAL: Preset voices are ALWAYS English, so set voice_language = "en" for preset voices
        # For user voices, try to get from input parameter, then auto-detect from stored metadata,
        # finally default to target language
        voice_language = inp.get("voice_language")
        
        voice_language_from_metadata = False
        
        # If using a preset voice, preset voices are ALWAYS English
        if preset_voice:
            voice_language = "en"
            print(f"[tts] Preset voice detected - setting voice_language to 'en' (preset voices are English-only)")
        elif voice_language is None and user_id and embedding_filename:
            # Auto-detect from stored metadata if not provided (only for user voices)
            stored_language = get_voice_language(user_id, embedding_filename)
            if stored_language:
                voice_language = stored_language
                voice_language_from_metadata = True
                print(f"[tts] Auto-detected voice language from metadata: {voice_language}")
        
        # Default to target language if still not found (only for user voices without metadata)
        if voice_language is None:
            voice_language = language
            print(f"[tts] Voice language not specified, defaulting to target language: {language}")
        
        # Determine if voice language was defaulted (not explicitly known)
        voice_language_was_auto_detected = (
            inp.get("voice_language") is None and 
            not preset_voice and 
            not voice_language_from_metadata
        )
        
        # Automatic accent control based on language matching:
        # - If voice_language == language: preserve voice accent (same language, keep natural accent)
        # - If voice_language != language: use target language accent (different language, prioritize target)
        # This can be overridden by explicitly setting preserve_voice_accent
        languages_match = voice_language == language
        
        # Determine if we should preserve the voice accent
        # Default: Yes, preserve it unless explicitly told otherwise or for preset voices across languages
        # This fixes the "unwanted accent" issue where it was too aggressively switching to target language accent
        if "preserve_voice_accent" in inp:
            # Explicitly set by user
            preserve_voice_accent = bool(inp.get("preserve_voice_accent", False))
        elif preset_voice:
            # For preset voices (always English initially), use target language accent if languages differ
            preserve_voice_accent = languages_match
        else:
            # Users voices: ALWAYS preserve voice accent by default (even across languages)
            # This prevents your English voice from sounding French when reading French text
            # Only switch if voice_language was completely unknown (no metadata)
            preserve_voice_accent = True
            if voice_language_was_auto_detected:
                preserve_voice_accent = False
        
        # Accent control parameter - controls how much to prioritize target language accent
        # vs voice sample accent (0.0 = use voice accent, 1.0 = prioritize target language accent)
        # CRITICAL: When voice_language was auto-detected or languages differ, ALWAYS use maximum accent control
        # to ensure target language accent is used (e.g., German accent for German text, even with English voice)
        if preserve_voice_accent:
            accent_control = 0.0
            print(f"[tts] ✅ Preserving voice accent (explicitly requested or languages match with explicit voice_language)")
        else:
            # Force maximum accent control when languages differ OR when voice_language was auto-detected
            # This ensures target language accent is always used
            accent_control = float(inp.get("accent_control", 1.0))
            if not languages_match or voice_language_was_auto_detected:
                print(f"[tts] ⚠️  Using target language accent ({language}) - accent_control={accent_control:.2f} (MAXIMUM)")
                print(f"[tts]   Reason: {'languages differ' if not languages_match else 'voice_language was auto-detected'}")
        
        # Volume normalization parameter - default to True for backward compatibility
        normalize_volume = inp.get("normalize_volume", True)

        # ===== OPTIMIZED PARAMETERS FOR CLEAN STORYTELLING AUDIO =====
        # These defaults match Resemble AI's official Multilingual demo for best quality
        # Balances voice cloning accuracy with clean, artifact-free audio generation
        
        # Exaggeration: Controls prosody expressiveness
        # 0.5 = neutral, natural storytelling (was 0.6 - too dramatic)
        # Lower = more natural pacing, fewer artifacts in emotional peaks
        exaggeration = float(inp.get("exaggeration", 0.5))
        
        # Temperature: Controls generation randomness/stability
        # 0.6 = Stable middle ground (was 0.55 - too low can cause noise/skipping)
        # Higher than before to prevent "mode collapse" where the model skips sentences
        temperature = float(inp.get("temperature", 0.6))
        
        # CFG Weight: 0.6 = Slightly more voice influence (was 0.5)
        # Helps the model stay "focused" on the voice to avoid generating noise
        cfg_weight = float(inp.get("cfg_weight", 0.6))

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
        
        # For multilingual model, always ensure consistent parameters for accent consistency
        # Even when languages match, we want consistent generation across all chunks
        # CRITICAL: Apply accent control if languages differ OR if voice_language was auto-detected
        # (auto-detected means we don't know the actual voice language, so we should use target accent)
        if model_type == "multilingual":
            # Apply accent control if: languages differ OR voice_language was auto-detected
            should_apply_accent_control = (voice_language != language) or voice_language_was_auto_detected
            
            if should_apply_accent_control:
                if preserve_voice_accent or accent_control < 0.1:
                    # Preserve voice accent: use original parameters to keep voice's natural accent
                    # No adjustments needed - let the voice embedding control the accent
                    print(f"[tts] Preserving voice accent: using original parameters to keep {voice_language} accent")
                    print(f"[tts] Parameters: temp={base_temperature:.2f}, cfg={base_cfg_weight:.2f}, exaggeration={base_exaggeration:.2f}")
                elif accent_control > 0.0:
                    # Extremely aggressive adjustments for consistent target language accent across all chunks
                    # Higher temperature = more variation = less voice accent influence
                    # With accent_control=1.0, maximize temperature to reduce voice accent as much as possible
                    base_temperature = min(temperature + (accent_control * 0.5), 1.0)
                    
                    # Lower cfg_weight = less voice embedding influence = more language model control
                    # With accent_control=1.0, minimize cfg_weight to absolute minimum to prioritize target language accent
                    # Reduce by 80% to strongly favor language model over voice embedding (clamped to minimum 0.1)
                    base_cfg_weight = max(cfg_weight * (1.0 - accent_control * 0.8), 0.1)
                    
                    # Reduce exaggeration more to make accent more consistent and prioritize target language
                    base_exaggeration = exaggeration * (1.0 - accent_control * 0.2)
                    
                    print(f"[tts] Accent control active (EXTREME strength for {language} accent): temp={base_temperature:.2f} (was {temperature:.2f}), "
                          f"cfg={base_cfg_weight:.2f} (was {cfg_weight:.2f}), "
                          f"exaggeration={base_exaggeration:.2f} (was {exaggeration:.2f})")
                    print(f"[tts] These parameters will be applied consistently to ALL chunks to ensure {language} accent")
            else:
                # Languages match - use consistent parameters for all chunks to ensure accent consistency
                # Lower temperature slightly to reduce variation and ensure consistent accent
                base_temperature = temperature * 0.95  # Slight reduction for more consistency
                print(f"[tts] Languages match - using consistent parameters for accent consistency across all chunks")
                print(f"[tts] Parameters: temp={base_temperature:.2f}, cfg={base_cfg_weight:.2f}, exaggeration={base_exaggeration:.2f}")
        
        # Use a consistent seed for ALL chunks to ensure accent consistency
        # Different seeds per chunk can cause accent variation
        consistent_seed = 42  # Fixed seed for maximum consistency
        torch.manual_seed(consistent_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(consistent_seed)
        print(f"[tts] Using consistent seed ({consistent_seed}) for all chunks to ensure accent consistency")
        
        # Store final parameters as constants to ensure they're never modified
        # These exact values will be used for EVERY chunk without any variation
        final_temperature = base_temperature
        final_cfg_weight = base_cfg_weight
        final_exaggeration = base_exaggeration
        
        # IMPORTANT: Disable potential randomness in model generation
        # Reset torch random state before the loop
        state = torch.get_rng_state()
        
        # Log the exact parameters that will be used for ALL chunks
        if model_type == "multilingual":
            print(f"[tts] LOCKED parameters for ALL chunks: temp={final_temperature:.6f}, cfg={final_cfg_weight:.6f}, exaggeration={final_exaggeration:.6f}, language_id={language}")
        else:
            print(f"[tts] LOCKED parameters for ALL chunks: temp={final_temperature:.6f}, cfg={final_cfg_weight:.6f}, exaggeration={final_exaggeration:.6f}")
        
        audio_chunks = []
        
        import time
        total_gen_start = time.time()

        for i, chunk_text in enumerate(chunks):
            chunk_start = time.time()
            print(f"[tts] Generating chunk {i+1}/{len(chunks)}: {chunk_text[:50]}...")
            try:
                # Use consistent seed (set before loop) for all chunks to ensure accent consistency
                # Re-set seed before each chunk to ensure deterministic generation
                # CRITICAL: Always reset the seed to the EXACT SAME value for EVERY chunk
                torch.manual_seed(consistent_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(consistent_seed)
                
                # Restore RNG state to ensure other random operations (if any) are consistent
                torch.set_rng_state(state)

                # Use inference_mode for faster inference (no gradient tracking)
                with torch.inference_mode():
                    # Different generate call based on model type
                    if model_type == "multilingual":
                        # CRITICAL: Use the EXACT same locked parameters for every chunk
                        # These values are calculated once and never modified to ensure consistency
                        output = tts.generate(
                            text=chunk_text,
                            audio_prompt_path=str(voice_path),
                            language_id=language,  # Same language_id for all chunks
                            exaggeration=final_exaggeration,  # Same exaggeration for all chunks
                            temperature=final_temperature,  # Same temperature for all chunks
                            cfg_weight=final_cfg_weight,  # Same cfg_weight for all chunks
                        )
                        # Verify parameters are being used correctly (log first chunk only to avoid spam)
                        if i == 0:
                            print(f"[tts] ✓ Chunk {i+1} using verified parameters: temp={final_temperature:.6f}, cfg={final_cfg_weight:.6f}, exaggeration={final_exaggeration:.6f}")
                    else:
                        # English model doesn't use language_id
                        # Use locked parameters for consistency (same for all chunks)
                        output = tts.generate(
                            text=chunk_text,
                            audio_prompt_path=str(voice_path),
                            exaggeration=final_exaggeration,  # Same for all chunks
                            temperature=final_temperature,  # Same for all chunks
                            cfg_weight=final_cfg_weight,  # Same for all chunks
                        )
                        # Verify parameters are being used correctly (log first chunk only to avoid spam)
                        if i == 0:
                            print(f"[tts] ✓ Chunk {i+1} using verified parameters: temp={final_temperature:.6f}, cfg={final_cfg_weight:.6f}, exaggeration={final_exaggeration:.6f}")

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

        final_wav = stitch_chunks(audio_chunks, chunks, pause_ms=pause_ms)

        # --- Apply comprehensive audio post-processing to reduce artifacts ---
        if len(final_wav) > 0:
            print(f"[tts] Applying audio post-processing to improve quality...")
            
            # Step 1: Remove DC offset (constant bias that can cause artifacts)
            final_wav = remove_dc_offset(final_wav)
            
            # Step 2: Detect and remove clicks/pops (sudden amplitude spikes)
            final_wav = detect_and_remove_clicks(final_wav, threshold=0.3)
            
            # Step 3: Apply smoothing to reduce rapid amplitude changes
            # REMOVED: Smoothing filter was causing muffled "pillow" sound - high frequencies were being dulled
            # final_wav = apply_smoothing_filter(final_wav, window_size=5)
            
            # Step 4: Apply high-pass filter to remove low-frequency artifacts (howling, rumble)
            # REMOVED: High-pass filter already applied during stitching (line 424) - duplicate application was over-filtering
            # print(f"[tts]   High-pass filter (100Hz) to remove low-frequency artifacts...")
            # final_wav = apply_high_pass_filter(final_wav, cutoff_hz=100)
            
            # Step 4.5: Apply high-shelf filter to gently reduce high-frequency noise
            # Reduces scratchy/sandy artifacts while preserving speech clarity
            print(f"[tts]   High-shelf filter to reduce high-frequency noise...")
            final_wav = apply_high_shelf_filter(final_wav, cutoff_hz=10000, gain_db=-2)
            
            # Step 5: Low-pass filter removed - was making audio sound too dimmed/muffled
            # High-shelf filter above is gentler and more effective
            
            # Step 6: Apply spectral gating to remove background noise and hiss
            # -60dB threshold: Highly aggressive to eliminate "shhhh" noise/hissing
            # This ensures that quiet sections and sentence tails are perfectly silent
            print(f"[tts]   Spectral gating to remove background noise...")
            final_wav = apply_spectral_gating(final_wav, threshold_db=-60)
            
            # Step 6.5: Reduce harsh breathing artifacts
            # Enhanced with ZCR detection and more aggressive reduction for cleaner pauses
            print(f"[tts]   Reducing breathing artifacts...")
            final_wav = reduce_breathing_artifacts(final_wav, breath_threshold_db=-35, reduction_factor=0.2)
            
            # Step 7: Apply smooth fade in/out to prevent clicks at start/end
            print(f"[tts]   Applying smooth fade in/out...")
            final_wav = apply_fade(final_wav, fade_ms=100)  # Longer fade for smoother edges
            
            print(f"[tts] Audio post-processing complete")

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
                if max_after > 0.90:
                    # Apply soft limiter instead of hard clipping for smoother peaks
                    final_wav = soft_limiter(final_wav, threshold=0.88, ratio=3.5)
                    new_peak = np.max(np.abs(final_wav))
                    print(f"[tts] Applied soft limiter to prevent clipping (peak: {max_after:.3f} → {new_peak:.3f})")
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
        elif action == "clone_voice_verified":
            return clone_voice_verified_handler(job)
        elif action == "verify_recording":
            return verify_recording_handler(job)
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