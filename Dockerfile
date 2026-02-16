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

# Consolidate pip installs to prevent redundant 1.3GB CUDA re-downloads
COPY requirements.txt .
RUN pip install --no-cache-dir \
    git+https://github.com/QwenLM/Qwen3-TTS.git \
    -r requirements.txt

# Copy the application code
COPY handler.py .

# The model is expected at /runpod-volume/qwen3_models (network volume)
RUN mkdir -p /runpod-volume/qwen3_models

# Start the handler directly (no start.sh needed)
CMD ["python", "-u", "handler.py"]
