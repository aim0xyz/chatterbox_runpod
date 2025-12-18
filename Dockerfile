FROM python:3.10-slim

WORKDIR /app

ENV MODEL_PATH="/models"
ENV RUNPOD_STORAGE_ROOT="/workspace"
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV HF_DATASETS_CACHE="/runpod-volume/.cache/huggingface/datasets"
ENV TRANSFORMERS_CACHE="/runpod-volume/.cache/huggingface/transformers"

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/resemble-ai/chatterbox.git .

RUN pip install --timeout=600 --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0

RUN pip install --no-cache-dir \
    "pydantic>=1.10.0,<2.0" \
    "spacy>=3.4.0,<3.5.0" \
    "gradio>=3.16,<3.30" \
    "typer>=0.3.0,<0.8.0"

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

RUN pip install --no-cache-dir --no-deps \
    resemble-perth==1.0.1 \
    pykakasi==2.3.0 \
    conformer==0.3.2 \
    spacy-pkuseg==1.0.1 \
    git+https://github.com/Vuizur/add-stress-to-epub@master#egg=russian-text-stresser

RUN pip install --no-cache-dir s3tokenizer

RUN pip install --no-cache-dir --no-deps -e .

RUN python -c "import chatterbox; print('Chatterbox installed')" || echo "Warning: Chatterbox check failed"

RUN pip install --no-cache-dir "runpod>=0.9.0" soundfile

COPY handler.py /handler.py
COPY start.sh /start.sh

RUN chmod +x /start.sh

CMD ["/start.sh"]