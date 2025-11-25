# Base Image: Use a specific version of Python on a slim Debian OS
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Set an environment variable for the model path
# RunPod network volumes are mounted at /workspace by default
ENV MODEL_PATH="/models"
ENV RUNPOD_STORAGE_ROOT="/workspace"

# Cache Hugging Face files on the volume to avoid filling container disk
# Note: TRANSFORMERS_CACHE is deprecated in Transformers v5, using HF_HOME instead
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV HF_DATASETS_CACHE="/runpod-volume/.cache/huggingface/datasets"
# Keep TRANSFORMERS_CACHE for compatibility with older versions but prioritize HF_HOME
ENV TRANSFORMERS_CACHE="/runpod-volume/.cache/huggingface/transformers"

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the Chatterbox source code from GitHub into the current directory
RUN git clone https://github.com/resemble-ai/chatterbox.git .

# --- INSTALL PYTORCH WITH CUDA 12.1 SUPPORT ---
# If your endpoint is CPU-only, switch to CPU wheels instead of cu121.
RUN pip install --timeout=600 --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0

# --- INSTALL COMPATIBLE VERSIONS TO RESOLVE CONFLICTS ---
# These pins imply Pydantic v1, so we must use FastAPI < 0.100 (v2 boundary).
RUN pip install --no-cache-dir \
    "pydantic>=1.10.0,<2.0" \
    "spacy>=3.4.0,<3.5.0" \
    "gradio>=3.16,<3.30" \
    "typer>=0.3.0,<0.8.0"

# Install other required dependencies with version constraints
# Using newer transformers version for better attention support
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
    onnxruntime==1.19.2

# Install remaining project dependencies (including russian-text-stresser from GitHub)
RUN pip install --no-cache-dir --no-deps \
    resemble-perth==1.0.1 \
    pykakasi==2.3.0 \
    conformer==0.3.2 \
    spacy-pkuseg==1.0.1 \
    git+https://github.com/Vuizur/add-stress-to-epub@master#egg=russian-text-stresser

# Install s3tokenizer WITH dependencies (it needs specific versions)
RUN pip install --no-cache-dir s3tokenizer

# Install Chatterbox in editable mode without dependencies (since we handled them manually)
RUN pip install --no-cache-dir --no-deps -e .

# Verify Chatterbox installation
RUN python -c "import chatterbox; print('Chatterbox version:', chatterbox.__version__ if hasattr(chatterbox, '__version__') else 'installed')" || \
    python -c "from chatterbox import tts; print('Chatterbox TTS module found')" || \
    python -c "from chatterbox.tts import ChatterboxTTS; print('ChatterboxTTS class available')" || \
    echo "Warning: Chatterbox verification failed but continuing..."

# Install the serverless handler HTTP stack
RUN pip install --no-cache-dir \
    "runpod>=0.9.0"

# Copy your custom handler into the container
COPY handler.py .

# Verify handler can at least be parsed
RUN python -c "import handler; print('Handler module loads successfully')" || echo "Warning: Handler import test failed"

# Command to run when the container starts
CMD ["python", "-u", "handler.py"]