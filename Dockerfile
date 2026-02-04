# Use NVIDIA CUDA 12.8 runtime image (latest CUDA 12.x with best optimizations)
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV PYTHONPATH="/app"

# Install base system dependencies first
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    gnupg \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Install Python 3.12 and remaining dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set python3.12 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and install build helpers
RUN python3 -m pip install --upgrade pip setuptools wheel

# 1. Install PyTorch with CUDA 12.4 support
RUN python3 -m pip install --no-cache-dir \
    torch \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 2. Flash Attention will be installed from volume in start.sh

# 3. Install remaining dependencies from requirements.txt
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Pre-download the speech tokenizer and common models (0.6B version for speed)
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-Base', filename='config.json')" || echo "Pre-download skipped"

# Copy the rest of the application code
COPY handler.py .
COPY start.sh .

# Ensure start.sh is executable
RUN chmod +x start.sh

# The model is expected at /qwen3_models
RUN mkdir -p /qwen3_models

# Run start.sh when the container launches
CMD ["/app/start.sh"]
