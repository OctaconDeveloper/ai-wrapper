# =============================================================================
# Multi-Model AI Platform — Dockerfile
#
# Single container running:
#   - ComfyUI (SDXL image generation) on port 8188 (internal)
#   - FastAPI (unified API gateway) on port 8000 (exposed)
#
# Managed by supervisord.
#
# Build:  docker build -t ai-multi-models .
# Run:    docker run --gpus all -p 8000:8000 -v ./model-cache:/workspace/models ai-multi-models
# =============================================================================

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ─────────────────────────────────────────────────────────────────
# 1. System dependencies
# ─────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip setuptools wheel

# ─────────────────────────────────────────────────────────────────
# 2. Create workspace structure
# ─────────────────────────────────────────────────────────────────
RUN mkdir -p /workspace/models/llm \
    /workspace/models/video \
    /workspace/models/audio \
    /workspace/outputs \
    /workspace/app-root \
    /var/log/supervisor

WORKDIR /workspace/app-root

# ─────────────────────────────────────────────────────────────────
# 3. Build Arguments & PyTorch Installation
# ─────────────────────────────────────────────────────────────────
ARG DEVICE=cpu
ENV DEVICE=${DEVICE}

RUN echo "Building for device: ${DEVICE}"

# Install PyTorch based on device
RUN if [ "$DEVICE" = "cuda" ]; then \
    python3 -m pip install --no-cache-dir torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124; \
    else \
    python3 -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ─────────────────────────────────────────────────────────────────
# 4. Install llama-cpp-python
# ─────────────────────────────────────────────────────────────────
RUN if [ "$DEVICE" = "cuda" ]; then \
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python3 -m pip install --no-cache-dir llama-cpp-python==0.3.8; \
    else \
    python3 -m pip install --no-cache-dir llama-cpp-python==0.3.8; \
    fi

# ─────────────────────────────────────────────────────────────────
# 5. Install Python dependencies
# ─────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────
# 6. Clone and install ComfyUI
# ─────────────────────────────────────────────────────────────────
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI \
    && cd /workspace/ComfyUI \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────
# 7. Copy application code
# ─────────────────────────────────────────────────────────────────
COPY app/ ./app/
COPY comfyui/ ./comfyui/
COPY scripts/ ./scripts/
COPY supervisord.conf ./supervisord.conf
COPY setup.sh ./setup.sh
COPY .env.example ./.env

# Make scripts executable
RUN chmod +x ./setup.sh ./scripts/download_models.sh ./scripts/run_comfyui.sh

# ─────────────────────────────────────────────────────────────────
# 8. Environment variables (defaults)
# ─────────────────────────────────────────────────────────────────
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV COMFYUI_PORT=8188
ENV COMFYUI_HOST=127.0.0.1
ENV MODELS_DIR=/workspace/models
ENV COMFYUI_DIR=/workspace/ComfyUI
ENV OUTPUT_DIR=/workspace/outputs
ENV MIXTRAL_MODEL_PATH=/workspace/models/llm/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf
ENV MIXTRAL_CONTEXT_LENGTH=32768
ENV MIXTRAL_GPU_LAYERS=-1
ENV WAN_T2V_MODEL=/workspace/models/video/wan2.1-t2v-1.3b
ENV WAN_I2V_MODEL=/workspace/models/video/wan2.1-i2v-14b
ENV XTTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
ENV MAX_LOADED_MODELS=3
ENV ENABLE_MODEL_OFFLOAD=true

# ─────────────────────────────────────────────────────────────────
# 9. Expose ports and set entrypoint
# ─────────────────────────────────────────────────────────────────
EXPOSE 8000 8188

# Entrypoint: downloads models (if needed) then starts supervisord
ENTRYPOINT ["bash", "./setup.sh"]
