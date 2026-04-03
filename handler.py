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
from typing import List, Union, Optional, Dict, Any, Tuple

# ==========================================
# IMPORT MODEL
# ==========================================
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel



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

# --- LAZY TURBO ACTIVATION (The Bucket Strategy) ---
was_turbo_baked = False

def ensure_turbo_activated():
    """Trigger the intensive Turbo Mode optimizations on the first request."""
    global model, was_turbo_baked
    if was_turbo_baked:
        return
    
    import shutil
    has_compiler = shutil.which("g++") or shutil.which("clang++") or os.environ.get("CC")
    if not has_compiler:
        print("[Turbo] ⚠️ Skipping Turbo: No C++ compiler found.")
        was_turbo_baked = True
        return

    print("[Turbo] 🚀 Activating Nitro-Engine (100+ tokens/s Mode)...")
    try:
        # 1. HARDWARE ACCELERATION (Safe & Fast)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # CUDA Graphs MUST be disabled for autoregressive TTS.
        # The KV cache grows dynamically during generation, which breaks
        # CUDA graph replay (requires fixed tensor shapes). This caused
        # the first request to be fast (43s) but subsequent ones 4x slower
        # (170s) due to re-recording or eager fallback overhead.
        # Triton kernel optimizations from max-autotune are still active.
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraphs = False
        
        # 2. NITRO COMPILATION
        # We only compile the main backbone. It's stable and fast.
        if hasattr(model.model, "talker"):
            print("[Turbo] 🏗️  Pre-compiling Talker Layers (Nitro - Max Autotune)...")
            # Use max-autotune for best Triton kernel performance with dynamic shapes
            # reduce-overhead often fails with dynamic KV cache growth
            model.model.talker = torch.compile(model.model.talker, mode="max-autotune")

        # 3. WARMUP (Master Bake)
        print("[Turbo] ⏳ Baking Master-Bucket (Full Generate)... Expect ~5 min hang here (ONCE EVER)...")
        dummy_ref = "/runpod-volume/preset_voices/en/Owen.wav"
        if not os.path.exists(dummy_ref):
            for f in PRESET_ROOT.rglob("*.wav"):
                dummy_ref = str(f)
                break
            
        if dummy_ref:
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Simple generation to ensure everything is in VRAM
                model.generate_voice_clone(
                    text="Engine warmup. Stability mode active.", 
                    language="english", 
                    ref_audio=dummy_ref,
                    ref_text="This is a warmup reference text.",
                    max_new_tokens=256  # Just enough to trigger compilation, not full generation
                 )
        print("[Turbo] ✅ Nitro-Engine Active. No more hangs, no more crashes.")
    except Exception as e:
        print(f"[Turbo] ⚠️ Warning: Turbo skipped for stability: {e}")
    
    was_turbo_baked = True

def init_model():
    """Fast load: Get the model on the GPU and report ACTIVE to RunPod."""
    global model, torch
    if model is not None:
        return
        
    if not hasattr(torch, "compiler"):
        class FakeCompiler:
            def is_compiling(self): return False
        torch.compiler = FakeCompiler()
    elif not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False

    try:
        from qwen_tts import Qwen3TTSModel
        
        # Use the persistent cache!
        cache_dir = "/runpod-volume/.torch_compile_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        
        print(f"[startup] Fast-Loading Qwen3: {MODEL_NAME_OR_PATH}...")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            import logging
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
            logging.getLogger("qwen_tts.core.models.configuration_qwen3_tts").setLevel(logging.ERROR)
        except: pass
        
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSConfig
        _old_talker_init = Qwen3TTSTalkerConfig.__init__
        def _patched_talker_init(self, *args, **kwargs):
            if "code_predictor_config" not in kwargs or kwargs["code_predictor_config"] is None:
                kwargs["code_predictor_config"] = Qwen3TTSTalkerCodePredictorConfig().to_dict()
            return _old_talker_init(self, *args, **kwargs)
        Qwen3TTSTalkerConfig.__init__ = _patched_talker_init

        # Explicitly load config to set torch_dtype for Flash Attention 2
        # This avoids "torch_dtype is deprecated" warning and ensures FA2 works
        config = Qwen3TTSConfig.from_pretrained(str(MODEL_NAME_OR_PATH))
        
        # Pass dtype and attn_implementation directly to from_pretrained
        # This is the modern, correct way to initialize
        model = Qwen3TTSModel.from_pretrained(
            str(MODEL_NAME_OR_PATH),
            config=config,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[VRAM] Clean load: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
        
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            model.tokenizer.padding_side = 'left'
            model.tokenizer.pad_token = model.tokenizer.eos_token

        print("[startup] Base model ready (Turbo will activate on FIRST request).")
    except Exception as e:
        print(f"[startup] Critical Error loading model: {e}")

# Call light init at startup
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
    """Preprocess text for expressive, naturally-paced children's storytelling.
    
    The TTS model uses punctuation as pacing cues — commas create short pauses,
    periods create longer pauses, and ellipses create dramatic breathing pauses.
    By strategically adding these cues, we guide the model to speak at a calm,
    unhurried pace WITHOUT any post-processing audio warping.
    
    This is the RIGHT way to control speech pace: at the text level, so the
    model naturally generates slower, more deliberate speech.
    """
    import re
    
    # 1. Normalize Unicode quotes and dashes to ASCII
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # smart double quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # smart single quotes
    text = text.replace('\u2014', ' — ')   # em-dash → keep as pause marker
    text = text.replace('\u2013', ' — ')   # en-dash → same
    text = text.replace('\u2026', '...')    # ellipsis character → three dots
    
    # 2. Ensure there is always a space after sentence-ending punctuation
    #    so the model perceives clear sentence boundaries
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # 3. Normalize multiple spaces / newlines into clean boundaries
    text = re.sub(r'\n\s*\n', '\n\n', text)   # normalize paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # single newline → space
    
    # 4. Paragraph breaks → ellipsis pause (model treats "..." as a natural breath)
    text = text.replace('\n\n', '... ')
    
    # 5. PACING CUES FOR CALM STORYTELLING:
    #    Insert a comma after long clauses (15+ words without punctuation)
    #    so the model takes a natural micro-breath. This is exactly how
    #    audiobook narrators pace themselves — breathing between thoughts.
    words = text.split()
    paced_words = []
    words_since_pause = 0
    for word in words:
        paced_words.append(word)
        words_since_pause += 1
        # Check if this word already ends with punctuation
        has_punctuation = word and word[-1] in '.!?,;:—-"\')'
        if has_punctuation:
            words_since_pause = 0
        elif words_since_pause >= 22:
            # Add a comma to create a breathing point — only every 22 words
            # (was 15, which was too frequent and sounded choppy/robotic)
            paced_words[-1] = word + ','
            words_since_pause = 0
    text = ' '.join(paced_words)
    
    # 6. EXPRESSIVENESS BOOST FOR KIDS STORIES:
    #    Exclamation points carry much more prosodic energy than periods.
    #    Convert obvious story exclamation sentences (short, energetic endings)
    #    to use '!' so the model narrates them with excitement and wonder.
    #    Only applies to sentences that are clearly exclamatory in nature.
    def boost_expressive_sentences(t):
        """Heuristically replace '.' with '!' on short, energetic sentences."""
        exclamatory_starters = (
            'Oh', 'Wow', 'Yes', 'No', 'Hooray', 'Yay', 'What', 'How wonderful',
            'How exciting', 'How magical', 'Look', 'Listen', 'Amazing', 'Incredible',
            'Suddenly', 'All of a sudden', 'At last', 'Finally', 'And then',
        )
        def replace_match(m):
            sentence = m.group(0)
            # Only apply to short sentences (< 80 chars) ending in period (not already !?)
            if sentence.endswith('.') and len(sentence) < 80:
                for starter in exclamatory_starters:
                    if sentence.startswith(starter):
                        return sentence[:-1] + '!'
            return sentence
        # Match individual sentences
        return re.sub(r'[A-Z][^.!?]*[.!?]', replace_match, t)
    
    text = boost_expressive_sentences(text)
    
    # 7. Collapse excessive whitespace
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def match_spectral_envelope(wavs, sr):
    """Match the spectral envelope of all chunks to the first chunk.
    
    Each chunk may have slightly different 'room tone' or frequency balance
    due to random sampling in the subtalker. This function applies a gentle
    spectral transfer so all chunks share the same frequency profile as
    the first chunk — making them sound like the same recording session.
    
    Uses a smoothed spectral ratio approach (like professional audio mastering)
    with soft clamping to avoid artifacts.
    """
    if len(wavs) < 2:
        return wavs
    
    from scipy.signal import welch
    from scipy.interpolate import interp1d
    
    def get_spectral_envelope(audio, sr):
        """Get smoothed spectral envelope using Welch's method."""
        nperseg = min(2048, len(audio) // 2)
        if nperseg < 256:
            return None, None
        freqs, psd = welch(audio, fs=sr, nperseg=nperseg)
        # Smooth to get envelope (not fine detail)
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(psd, size=max(5, len(psd) // 20))
        return freqs, smoothed
    
    # Get reference spectrum from the first chunk
    ref_freqs, ref_spectrum = get_spectral_envelope(wavs[0], sr)
    if ref_freqs is None:
        return wavs
    
    matched = [wavs[0]]  # First chunk is the reference
    
    for i, wav in enumerate(wavs[1:], 1):
        try:
            chunk_freqs, chunk_spectrum = get_spectral_envelope(wav, sr)
            if chunk_freqs is None:
                matched.append(wav)
                continue
            
            # Compute spectral ratio (how much to adjust each frequency band)
            # Avoid division by zero
            ratio = np.sqrt((ref_spectrum + 1e-10) / (chunk_spectrum + 1e-10))
            
            # Soft-clamp ratio to avoid extreme adjustments (max ±3dB per band)
            ratio = np.clip(ratio, 0.7, 1.4)
            
            # Apply the spectral correction via FFT
            fft = np.fft.rfft(wav)
            fft_freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
            
            # Interpolate the ratio to match FFT frequency bins
            ratio_interp = interp1d(
                chunk_freqs, ratio, 
                kind='linear', 
                bounds_error=False, 
                fill_value=1.0
            )(fft_freqs)
            
            # Apply correction
            fft_corrected = fft * ratio_interp
            wav_corrected = np.fft.irfft(fft_corrected, n=len(wav)).astype(np.float32)
            
            matched.append(wav_corrected)
            print(f"[spectral_match] 🎛️  Chunk {i}: matched to reference (ratio range: {ratio.min():.2f}-{ratio.max():.2f})")
        except Exception as e:
            print(f"[spectral_match] ⚠️  Chunk {i}: skipped ({e})")
            matched.append(wav)
    
    return matched


def soft_clip(audio, threshold_dbfs=-2.0):
    """Gently saturate the signal as it approaches 0 dBFS.
    
    Hard-clipping (np.clip) creates square-wave-like edges that sound like
    harsh digital distortion. Soft-clipping uses a sigmoid curve (tanh)
    to gently round off peaks that exceed the threshold, preserving
    the professional 'analog' warmth of the voice.
    """
    threshold = 10 ** (threshold_dbfs / 20.0)
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    if not np.any(mask):
        return audio
    
    # Standard soft-clipping: threshold + (scale * tanh((x-threshold)/scale))
    # This ensures the signal never actually hits 1.0 but gets very close.
    scale = 1.0 - threshold
    audio[mask] = np.sign(audio[mask]) * (threshold + scale * np.tanh((abs_audio[mask] - threshold) / scale))
    return audio


def normalize_audio_loudness(audio, sr, target_dbfs=-16.0):
    """Normalize audio loudness using RMS-based gain.
    
    Applies gain to bring the audio to a consistent loudness level.
    Target is -16 dBFS (slightly conservative) to leave headroom for the
    peak limiter that runs afterward on the final combined audio.
    """
    if len(audio) == 0:
        return audio
    
    # Calculate current RMS
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-8:  # Near-silent audio
        print(f"[normalize] ⚠️  Audio is near-silent (RMS={rms:.6f}), skipping normalization")
        return audio
    
    # Calculate target RMS from dBFS
    target_rms = 10 ** (target_dbfs / 20.0)  # -16 dBFS ≈ 0.1585
    
    # Calculate required gain
    gain = target_rms / rms
    
    # Cap gain to avoid amplifying noise excessively
    max_gain = 15.0
    if gain > max_gain:
        print(f"[normalize] ⚠️  Gain capped at {max_gain}x (original would be {gain:.1f}x)")
        gain = max_gain
    
    print(f"[normalize] 🔊 RMS: {rms:.4f} → target {target_rms:.4f} (gain={gain:.2f}x)")
    
    # Apply gain
    normalized = audio * gain
    
    # Soft-clip instead of hard-clip to prevent distortion
    normalized = soft_clip(normalized, threshold_dbfs=-1.5).astype(np.float32)
    
    final_rms = float(np.sqrt(np.mean(normalized ** 2)))
    print(f"[normalize] ✅ Final RMS: {final_rms:.4f} ({20 * np.log10(final_rms + 1e-10):.1f} dBFS)")
    
    return normalized


def apply_peak_limiter(audio, threshold_dbfs=-1.0, release_ms=50, sr=24000):
    """Soft-knee peak limiter to prevent clipping and loudness spikes.
    
    This version uses a fast attack (0.5ms) and a smooth release to bring
    audio levels under the threshold without the harsh 'clicking' or
    'pumping' sounds associated with instant-attack hard limiters.
    
    Args:
        audio: numpy float32 array of audio samples
        threshold_dbfs: ceiling level in dBFS (default -1.0 dBFS)
        release_ms: how quickly gain recovers after a peak
        sr: sample rate
    """
    if len(audio) == 0:
        return audio
    
    threshold_linear = 10 ** (threshold_dbfs / 20.0)
    
    # Check if limiting is even needed
    peak = float(np.max(np.abs(audio)))
    if peak <= threshold_linear:
        print(f"[limiter] ✅ No limiting needed (peak={20*np.log10(peak+1e-10):.1f} dBFS)")
        return audio
    
    print(f"[limiter] 🎚️  Applying peak limiter (peak={20*np.log10(peak+1e-10):.1f} dBFS → ceiling={threshold_dbfs} dBFS)")
    
    # Pre-calculate smoothing coefficients
    attack_ms = 0.5
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))
    
    abs_audio = np.abs(audio)
    gain_envelope = np.ones(len(audio), dtype=np.float32)
    current_gain = 1.0
    
    # Loop over samples applying smoothed gain reduction
    for i in range(len(audio)):
        # Calculate the instant gain required to hit the threshold
        target_gain = min(1.0, threshold_linear / max(abs_audio[i], 1e-10))
        
        if target_gain < current_gain:
            # Fast Engagement (Attack)
            current_gain = current_gain * attack_coeff + target_gain * (1.0 - attack_coeff)
        else:
            # Smooth Recovery (Release)
            current_gain = current_gain * release_coeff + target_gain * (1.0 - release_coeff)
            
        gain_envelope[i] = current_gain
    
    # Apply the gain reduction
    limited = (audio * gain_envelope).astype(np.float32)
    
    # Final safety soft-clip (should be minimal after limiter)
    limited = soft_clip(limited, threshold_dbfs=-0.2)
    
    final_peak = float(np.max(np.abs(limited)))
    print(f"[limiter] ✅ Limited audio peak: {20*np.log10(final_peak+1e-10):.1f} dBFS")
    
    return limited


def trim_internal_silence(audio, sr, max_silence_sec=0.35, silence_threshold_db=-40):
    """Trim excessively long silences WITHIN a generated audio chunk.
    
    The TTS model sometimes generates 1-2+ second silence gaps between sentences
    (especially after periods). This detects long silent regions and caps them
    at max_silence_sec (default 0.35s), producing natural-sounding pauses.
    
    Args:
        audio: numpy array of audio samples
        sr: sample rate
        max_silence_sec: maximum allowed silence duration (seconds)
        silence_threshold_db: threshold below which audio is considered silent (dB)
    
    Returns:
        Trimmed audio numpy array
    """
    if len(audio) < sr * 0.1:  # Skip very short audio
        return audio
    
    # Convert threshold from dB to linear amplitude
    silence_threshold = 10 ** (silence_threshold_db / 20.0)
    max_silence_samples = int(max_silence_sec * sr)
    
    # Use a frame-based approach for efficiency
    frame_size = int(0.02 * sr)  # 20ms frames
    
    # Calculate RMS energy per frame
    n_frames = len(audio) // frame_size
    if n_frames < 3:
        return audio
    
    is_silent = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        is_silent[i] = rms < silence_threshold
    
    # Find runs of silent frames
    result_parts = []
    i = 0
    audio_pos = 0
    trimmed_count = 0
    
    while i < n_frames:
        if is_silent[i]:
            # Count consecutive silent frames
            silence_start = i
            while i < n_frames and is_silent[i]:
                i += 1
            silence_end = i
            
            silence_sample_start = silence_start * frame_size
            silence_sample_end = min(silence_end * frame_size, len(audio))
            silence_duration_samples = silence_sample_end - silence_sample_start
            
            # Add any audio before this silence region
            if silence_sample_start > audio_pos:
                result_parts.append(audio[audio_pos:silence_sample_start])
            
            # If silence is too long, cap it
            if silence_duration_samples > max_silence_samples:
                result_parts.append(np.zeros(max_silence_samples, dtype=audio.dtype))
                trimmed_duration = (silence_duration_samples - max_silence_samples) / sr
                trimmed_count += 1
            else:
                # Keep the silence as-is
                result_parts.append(audio[silence_sample_start:silence_sample_end])
            
            audio_pos = silence_sample_end
        else:
            i += 1
    
    # Add remaining audio
    if audio_pos < len(audio):
        result_parts.append(audio[audio_pos:])
    
    if trimmed_count > 0:
        trimmed_audio = np.concatenate(result_parts) if result_parts else audio
        saved_sec = (len(audio) - len(trimmed_audio)) / sr
        print(f"[silence_trim] ✂️  Trimmed {trimmed_count} long silence(s), saved {saved_sec:.2f}s")
        return trimmed_audio
    
    return audio




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
        target_rms = 0.20  # Standard target for clear voice reference
        
        # Always normalize if it's too quiet OR too loud (e.g. RMS < 0.18 or > 0.22)
        # This ensures CONSISTENT volume for all voices
        if abs(rms - target_rms) > 0.02: 
            print(f"[ref_norm] � Normalizing reference (current RMS={rms:.4f} → target {target_rms})")
            
            # Calculate gain but cap it to avoid noise explosion on silent clips
            gain = target_rms / max(rms, 1e-8)
            gain = min(gain, 15.0) 
            
            # Apply gain and use soft-clipping to prevent harshness in the reference
            audio_np = audio_np * gain
            audio_np = soft_clip(audio_np, threshold_dbfs=-0.5).astype(np.float32)
            
            # Save normalized reference
            import torch
            normalized_waveform = torch.from_numpy(audio_np).unsqueeze(0)
            temp_path = "/tmp/normalized_ref.wav"
            torchaudio.save(temp_path, normalized_waveform, sample_rate_ref)
            print(f"[ref_norm] ✅ Reference normalized & clipped (gain={gain:.2f}x)")
            return temp_path
        else:
            print(f"[ref_norm] ✅ Reference audio level perfect (RMS={rms:.4f})")
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
    ensure_turbo_activated()
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
    
    # --- VOICE SETTINGS (CHILDREN'S STORYTELLING) ---
    # Balanced for natural, calm, expressive narration.
    # The text preprocessor adds pacing cues (commas, ellipses) that make the
    # model naturally speak at a relaxed pace. These params control expressiveness.
    #
    # Main Talker: Controls WHAT is said (prosody, word-level expression)
    #   - Temperature 0.85 → good pitch variety without being erratic
    #   - Top_p 0.90 → expressive but controlled inflections
    #   - Top_k 50 → standard vocabulary
    #   - Repetition_penalty 1.02 → gentle nudge against monotone repetition
    # --- GENERATION PARAMETERS (EXPRESSIVE CHILDREN'S STORYTELLING) ---
    # Main Talker: Controls WHAT is said — prosody, word-level expression, pitch variety.
    #   - Temperature 0.92 → more pitch variety and emotion than default 0.85,
    #     giving the narrator a natural storytelling lilt without being erratic
    #   - Top_p 0.95 → wider token selection allows richer intonation patterns
    #   - Top_k 50 → standard vocabulary breadth
    #   - Repetition_penalty 1.05 → stronger nudge against monotone repetition
    # Main talker: controls prosody, pitch variety, emotional inflection.
    # Higher temperature → more natural storytelling lilt and expressive range.
    # top_p 0.97 gives wider intonation patterns for a warmer, less robotic feel.
    temperature = float(inp.get("temperature", 0.97))
    top_p = float(inp.get("top_p", 0.97))
    repetition_penalty = float(inp.get("repetition_penalty", 1.08))
    top_k = int(inp.get("top_k", 60))
    
    # Subtalker (acoustic code predictor): Controls HOW it sounds — breath, tone, warmth.
    # More randomness here = organic variation between sentences (like a real narrator
    # naturally breathing and shifting slightly). Too low = robotically identical tone.
    # Too high = each chunk sounds like a different person.
    # Sweet spot for kids storytelling: temp 0.40, top_k 20, top_p 0.70
    subtalker_temperature = float(inp.get("subtalker_temperature", 0.40))
    subtalker_top_k = int(inp.get("subtalker_top_k", 20))
    subtalker_top_p = float(inp.get("subtalker_top_p", 0.70))
    subtalker_repetition_penalty = float(inp.get("subtalker_repetition_penalty", 1.08))
    
    if not text:
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
    # Larger chunks = fewer stitching points = more consistent audio
    text_chunks = split_into_natural_chunks(text, max_chars=400)
    
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

        # BATCH GENERATION: Process all chunks in parallel for maximum speed.
        # Per-chunk normalization + crossfading smooths out any tone differences.
        #
        # CRITICAL: Dynamic max_new_tokens based on actual text length.
        # In batch mode, ALL chunks generate for max_new_tokens steps (even if
        # some hit EOS early). One runaway chunk forces the entire batch to wait.
        # At 12Hz codec and ~120 wpm storytelling pace: 1 char ≈ 0.09s ≈ 1.1 tokens.
        # We add generous 80% margin so the model has room for natural pauses,
        # dramatic beats, and breathing — key for NOT sounding rushed.
        max_chunk_chars = max(len(c) for c in text_chunks)
        limit = min(max(int(max_chunk_chars * 1.1 * 1.8), 256), 768)
        
        # SORT BY LENGTH: Minimizes left-padding disparity in the batch.
        # Without this, a 100-char chunk padded to 400 chars gets shifted
        # positional encodings that alter voice timbre and expressiveness.
        # We sort before generation and un-sort after to preserve original order.
        sort_indices = sorted(range(len(text_chunks)), key=lambda i: len(text_chunks[i]))
        unsort_indices = [0] * len(sort_indices)
        for new_pos, old_pos in enumerate(sort_indices):
            unsort_indices[old_pos] = new_pos
        sorted_chunks = [text_chunks[i] for i in sort_indices]
        
        print(f"[TTS] 🚀 Starting BATCH processing of {len(text_chunks)} chunks...")
        print(f"[TTS] 📊 Dynamic token limit: {limit} (based on longest chunk: {max_chunk_chars} chars)")
        print(f"[TTS] 🔧 Consistency settings: subtalker_temp={subtalker_temperature}, subtalker_top_k={subtalker_top_k}, subtalker_top_p={subtalker_top_p}")
        
        gen_kwargs = dict(
            text=sorted_chunks,   # Sorted by length for minimal padding disparity
            language=language,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            max_new_tokens=limit,
            subtalker_dosample=True,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_repetition_penalty=subtalker_repetition_penalty,
            # CONSISTENCY FIX: non_streaming_mode gives the model full text context
            # before generating audio, resulting in more consistent prosody across chunks
            non_streaming_mode=True,
        )
        
        if voice_clone_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = voice_clone_prompt
        else:
            gen_kwargs["ref_audio"] = str(voice_path)
            gen_kwargs["ref_text"] = None
            gen_kwargs["x_vector_only_mode"] = True
        
        # Sync GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t0 = time.time()
        
        # Use a time-based seed so each story generation has natural variation.
        # A fixed seed (42) caused the same pitch contours every time → robotic.
        # A random seed lets the model breathe differently each time, like a real narrator.
        _seed = int(time.time() * 1000) % (2**31)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        print(f"[TTS] 🎲 Using generation seed: {_seed} (natural variation mode)")
        
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            wavs, sample_rate = model.generate_voice_clone(**gen_kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        batch_time = time.time() - t0
        print(f"[PROFILER] 🚀 Batch Generation Complete in {batch_time:.2f}s")
        
        # Un-sort wavs back to original chunk order
        wavs_unsorted = [None] * len(wavs)
        for sorted_pos, wav in enumerate(wavs):
            original_pos = sort_indices[sorted_pos]
            wavs_unsorted[original_pos] = wav
        wavs = wavs_unsorted
        
        # Process results: normalize each chunk + crossfade for seamless transitions
        sr = sample_rate
        CROSSFADE_MS = 80  # 80ms crossfade — wider overlap masks stitching seams
        crossfade_samples = int(CROSSFADE_MS / 1000.0 * sr)
        
        # Per-chunk post-processing: duration cap + silence trim + normalize
        processed_wavs = []
        for idx, (wav, chunk_text) in enumerate(zip(wavs, text_chunks)):
            # SKIP-SENTENCE FIX: Guard against None or empty/near-empty wavs.
            # When the model returns an empty or extremely short result for a chunk,
            # we synthesise a short silence placeholder so the story audio is
            # never truncated — and we log a warning so it's easy to diagnose.
            if wav is None or len(wav) == 0:
                print(f"[duration_cap] ⚠️  Chunk {idx} returned EMPTY audio — replacing with silence. Text: '{chunk_text[:60]}'")
                wav = np.zeros(int(0.5 * sr), dtype=np.float32)  # 500ms placeholder silence
                processed_wavs.append(wav)
                continue
            
            min_expected_samples = int(0.1 * sr)  # At least 100ms of real audio
            if len(wav) < min_expected_samples:
                print(f"[duration_cap] ⚠️  Chunk {idx} too short ({len(wav)} samples) — may be skipped audio. Text: '{chunk_text[:60]}'")
            
            # DURATION CAP: Truncate garbage audio at end of chunk.
            # The model sometimes generates noise/artifacts past the end of
            # meaningful speech ("farting" sounds). These have energy so the
            # silence trimmer can't catch them. We cap based on expected speech
            # duration: ~90ms per character + 100% margin for natural pauses,
            # breath-taking, and storytelling emphasis.
            max_expected_sec = max(len(chunk_text) * 0.09 * 2.0, 4.0)  # generous: at least 4s
            max_samples = int(max_expected_sec * sr)
            if len(wav) > max_samples:
                trimmed_sec = (len(wav) - max_samples) / sr
                print(f"[duration_cap] ✂️  Chunk {idx}: trimmed {trimmed_sec:.1f}s of excess audio (text: {len(chunk_text)} chars, max: {max_expected_sec:.1f}s)")
                # Apply a short fade-out (50ms) before truncating to avoid clicks
                fade_samples = min(int(0.05 * sr), max_samples)
                wav = wav[:max_samples].copy()
                wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples).astype(np.float32)
            
            # Trim excessively long silences WITHIN each chunk
            # (model sometimes generates 1-2s gaps between sentences)
            wav = trim_internal_silence(wav, sr, max_silence_sec=0.35)
            processed_wavs.append(wav)
        
        normalized_wavs = []
        for idx, wav in enumerate(processed_wavs):
            # Normalizing to -17 dBFS gives us plenty of headroom for crossfading
            # and prevents constructive interference from clipping.
            wav = normalize_audio_loudness(wav, sr, target_dbfs=-17.0)
            duration = len(wav) / sr
            char_count = len(text_chunks[idx].replace(' ', ''))
            rate = char_count / duration if duration > 0 else 0
            print(f"[chunk {idx}] 📊 {duration:.1f}s | {rate:.1f} chars/sec | {len(text_chunks[idx])} chars")
            normalized_wavs.append(wav)
        
        # SPECTRAL MATCHING: Make all chunks sound like the same recording session.
        # Even with consistent voice DNA, each chunk may have slightly different
        # room tone / frequency balance due to subtalker sampling. This gently
        # matches each chunk's spectral envelope to the first chunk's.
        try:
            normalized_wavs = match_spectral_envelope(normalized_wavs, sr)
            print(f"[TTS] 🎛️  Spectral matching complete — all chunks matched to reference")
        except Exception as e:
            print(f"[TTS] ⚠️  Spectral matching skipped: {e}")
        
        # Stitch chunks together with crossfade + pauses
        final_parts = [all_audio[0]]  # leading silence already added
        
        for i, (wav, chunk_text) in enumerate(zip(normalized_wavs, text_chunks)):
            if i == 0:
                final_parts.append(wav)
            else:
                # Crossfade: blend the end of previous chunk with start of current chunk
                prev = final_parts[-1]
                xf = min(crossfade_samples, len(prev), len(wav))
                
                if xf > 0:
                    # Create linear fade curves
                    fade_out = np.linspace(1.0, 0.0, xf).astype(np.float32)
                    fade_in = np.linspace(0.0, 1.0, xf).astype(np.float32)
                    
                    # Blend the overlap region
                    blended = prev[-xf:] * fade_out + wav[:xf] * fade_in
                    
                    # Replace end of previous with the blended part
                    final_parts[-1] = prev[:-xf]
                    final_parts.append(blended)
                    final_parts.append(wav[xf:])
                else:
                    final_parts.append(wav)
            
            # Add pause between chunks (but not after the last one)
            if i < len(text_chunks) - 1:
                pause = get_pause_duration_after_chunk(chunk_text, sr)
                final_parts.append(pause)
        
        # Combine all pieces
        final_audio = np.concatenate(final_parts)
        
        # --- SAFETY TRUNCATION ---
        # Prevent runaway generation from causing 400 Bad Request (Payload too large)
        # If audio is > 10 minutes, something is wrong with the model (loops).
        max_duration = 600.0 # 10 minutes
        current_duration = len(final_audio) / sr
        if current_duration > max_duration:
             print(f"[TTS] ⚠️  WARNING: Audio duration {current_duration:.2f}s exceeds limit {max_duration}s. Truncating.")
             # Cut to limit
             final_audio = final_audio[:int(max_duration * sr)]
             current_duration = max_duration
        
        # Step 1: Normalize the combined audio to a consistent loudness level.
        # We target -16.5 dBFS to leave enough headroom for peak limiting.
        print(f"[TTS] 🔊 Normalizing final audio loudness...")
        final_audio = normalize_audio_loudness(final_audio, sr, target_dbfs=-16.5)
        
        # Step 2: Peak limiter — prevents clipping and 'too loud' spikes.
        # Uses 0.5ms attack and soft-knee saturation for a professional finish.
        print(f"[TTS] 🎚️  Applying peak limiter (ceiling=-1.5 dBFS)...")
        final_audio = apply_peak_limiter(final_audio, threshold_dbfs=-1.5, release_ms=60, sr=sr)
        
        # WARMTH EQ: Gentle high-shelf boost (+1.5 dB above 6kHz) adds "air" and
        # presence — making the voice sound less flat/digital and more like a real
        # narrator recorded in a warm room. Kids respond well to this rounded tone.
        try:
            from scipy.signal import butter, sosfilt
            # High-shelf: boost frequencies above 6kHz by ~1.5dB
            nyq = sr / 2.0
            sos_hi = butter(2, 6000 / nyq, btype='high', output='sos')
            hi_shelf = sosfilt(sos_hi, final_audio)
            final_audio = (final_audio + 0.18 * hi_shelf).astype(np.float32)
            # Low-shelf: very gentle cut below 100Hz removes rumble/boominess
            sos_lo = butter(2, 100 / nyq, btype='high', output='sos')
            final_audio = sosfilt(sos_lo, final_audio).astype(np.float32)
            # Re-normalize after EQ to stay at target loudness
            final_audio = normalize_audio_loudness(final_audio, sr, target_dbfs=-16.5)
            final_audio = soft_clip(final_audio, threshold_dbfs=-0.5).astype(np.float32)
            print(f"[TTS] 🎛️  Warmth EQ applied (hi-shelf +1.5dB @6kHz, lo-cut @100Hz)")
        except Exception as e:
            print(f"[TTS] ⚠️  Warmth EQ skipped: {e}")
        
        duration = len(final_audio) / sr
        
        # 3. Encoding to MP3
        log_progress("convert", 92, "Finishing story audio...")
        
        wav_io = io.BytesIO()
        sf.write(wav_io, final_audio, sr, format='WAV')
        wav_io.seek(0)
        
        audio_segment = AudioSegment.from_wav(wav_io)
        mp3_io = io.BytesIO()
        # Reduced bitrate to 128k to fit within API payload limits for long stories (prevents timeout)
        audio_segment.export(mp3_io, format="mp3", bitrate="128k")
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
        import traceback
        print(f"Generation error: {e}")
        traceback.print_exc()
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
    # This activates the 5-minute Turbo optimization only on the first request 
    # to avoid hitting RunPod's 5-minute startup timeout.
    ensure_turbo_activated()
    
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
