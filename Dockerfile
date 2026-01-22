FROM python:3.10-slim

WORKDIR /app

ENV MODEL_PATH="/models"
ENV RUNPOD_STORAGE_ROOT="/workspace"
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV HF_DATASETS_CACHE="/runpod-volume/.cache/huggingface/datasets"
ENV TRANSFORMERS_CACHE="/runpod-volume/.cache/huggingface/transformers"

# Install system dependencies including audio libraries
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone Chatterbox
RUN git clone https://github.com/resemble-ai/chatterbox.git .

# Install PyTorch with CUDA support
RUN pip install --timeout=600 --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0

# Install core dependencies
RUN pip install --no-cache-dir \
    "pydantic>=1.10.0,<2.0" \
    "spacy>=3.4.0,<3.5.0" \
    "gradio>=3.16,<3.30" \
    "typer>=0.3.0,<0.8.0"

# Install Chatterbox dependencies
RUN pip install --timeout=600 --no-cache-dir \
    transformers==4.47.0 \
    diffusers==0.29.0 \
    safetensors==0.5.3 \
    librosa==0.11.0 \
    numpy==1.24.0 \
    einops==0.6.1 \
    "huggingface-hub>=0.23.2,<1.0" \
    requests==2.31.0 \
    packaging==23.1 \
    python-multipart==0.0.2 \
    onnx==1.17.0 \
    onnxruntime==1.19.2 \
    omegaconf>=2.3.0

# Install additional dependencies (no deps to avoid conflicts)
RUN pip install --no-cache-dir --no-deps \
    resemble-perth==1.0.1 \
    pykakasi==2.3.0 \
    conformer==0.3.2 \
    spacy-pkuseg==1.0.1 \
    git+https://github.com/Vuizur/add-stress-to-epub@master#egg=russian-text-stresser

RUN pip install --no-cache-dir s3tokenizer

# Install Chatterbox
RUN pip install --no-cache-dir --no-deps -e .

# Verify Chatterbox installation
RUN python -c "import chatterbox; print('Chatterbox installed')" || echo "Warning: Chatterbox check failed"

# ============================================
# NEW: Install Voice Verification Dependencies
# ============================================

# Core verification libraries
RUN pip install --no-cache-dir \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    joblib==1.3.2

# Speech-to-Text (faster-whisper) - use compatible versions
RUN pip install --no-cache-dir \
    faster-whisper==1.0.3

# Audio processing utilities
RUN pip install --no-cache-dir \
    pydub==0.25.1 \
    webrtcvad==2.0.10 \
    audioread==3.0.1

# ============================================
# AI Audio Enhancement (Resemble Enhance)
# ============================================

# Install Resemble Enhance for artifact removal
# This adds ~500MB to image size but significantly improves audio quality
RUN pip install --no-cache-dir resemble-enhance

# Pre-download enhancement models to avoid cold start
RUN python -c "from resemble_enhance.enhancer.inference import denoise, enhance; print('Resemble Enhance models cached')" || echo "Enhancement models will download on first use"

# Install RunPod and audio I/O
RUN pip install --no-cache-dir "runpod>=0.9.0" soundfile

# Pre-download Whisper model to avoid cold starts
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', download_root='/runpod-volume/.cache/whisper')" || echo "Whisper model will download on first use"

# Pre-download Chinese segmentation model to avoid cold start downloads
RUN python -c "from spacy_pkuseg import pkuseg; pkuseg()" || echo "pkuseg model will download on first use"

# Copy handler and startup script
COPY handler.py /handler.py
COPY start.sh /start.sh

RUN chmod +x /start.sh

CMD ["/start.sh"]