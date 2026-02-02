# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Hugging Face and RunPod persistence
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/runpod-volume/.cache/huggingface/transformers"
ENV PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (Optimized for RunPod GPUs)
RUN pip install --no-cache-dir \
    torch>=2.2.0 \
    torchaudio>=2.2.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install the core Qwen-TTS package to save time on setup
RUN pip install --no-cache-dir qwen-tts

# Pre-download the speech tokenizer and common models to avoid cold starts
# (Optional but recommended for production)
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', filename='config.json')" || echo "Pre-download skipped"

# Copy the rest of the application code
COPY handler.py .
COPY start.sh .

# Ensure start.sh is executable
RUN chmod +x start.sh

# The model is expected at /qwen3_models (mapped via RunPod Volume)
# Ensure the folder exists in the container just in case
RUN mkdir -p /qwen3_models

# Run start.sh when the container launches
CMD ["/app/start.sh"]
