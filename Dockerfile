# =====================================================
# Qwen3-TTS RunPod Serverless — Optimized Dockerfile
# Uses PyTorch native SDPA (no flash-attn compilation!)
# Build time: ~2 minutes (download only, no C++ compile)
# =====================================================
# Using PyTorch 2.4.0 for significantly better torch.compile stability and speed
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    git-lfs \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pre-install Flash Attention 2 from pre-built wheel (Saves 20+ mins build time)
# Using v2.6.3 for PyTorch 2.4.0 + CUDA 12.x + Python 3.11
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Consolidate pip installs
COPY requirements.txt .
RUN pip install --no-cache-dir \
    git+https://github.com/aim0xyz/qwen3-tts_aimoxyz.git \
    -r requirements.txt

# Copy the application code
COPY handler.py .

# The model is expected at /runpod-volume/qwen3_models (network volume)
RUN mkdir -p /runpod-volume/qwen3_models

# Start the handler directly (no start.sh needed)
CMD ["python", "-u", "handler.py"]
