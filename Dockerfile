# Use NVIDIA CUDA devel image to provide nvcc for building flash-attn
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV PYTHONPATH="/app"

# Install Python and core system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    git-lfs \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    curl \
    ninja-build \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install build helpers
RUN python3 -m pip install --upgrade pip setuptools wheel packaging ninja

# 1. Install PyTorch with CUDA support (MUST BE FINISHED BEFORE FLASH ATTN)
RUN python3 -m pip install --no-cache-dir \
    torch>=2.2.0 \
    torchaudio>=2.2.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 2. Install Flash Attention (Using pre-compiled binaries to avoid 30-min timeout)
# We point to the official CUDA 12.1 wheels
RUN python3 -m pip install flash-attn --no-cache-dir --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases/latest

# 3. Copy and install the rest of requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Pre-download the speech tokenizer and common models
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', filename='config.json')" || echo "Pre-download skipped"

# Copy the rest of the application code
COPY handler.py .
COPY start.sh .

# Ensure start.sh is executable
RUN chmod +x start.sh

# The model is expected at /qwen3_models
RUN mkdir -p /qwen3_models

# Run start.sh when the container launches
CMD ["/app/start.sh"]
