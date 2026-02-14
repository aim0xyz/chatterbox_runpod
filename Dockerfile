# =====================================================
# Qwen3-TTS RunPod Serverless — Optimized Dockerfile
# Uses PyTorch native SDPA (no flash-attn compilation!)
# Build time: ~2 minutes (download only, no C++ compile)
# =====================================================
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

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

# Install qwen-tts from source
RUN pip install --no-cache-dir git+https://github.com/QwenLM/Qwen3-TTS.git

# Install remaining dependencies from requirements.txt
# IMPORTANT: No flash-attn here — we use SDPA instead
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY handler.py .

# The model is expected at /runpod-volume/qwen3_models (network volume)
RUN mkdir -p /runpod-volume/qwen3_models

# Start the handler directly (no start.sh needed)
CMD ["python", "-u", "handler.py"]
