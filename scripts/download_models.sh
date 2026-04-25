#!/bin/bash
# =============================================================================
# Model Download Script — downloads all model weights at container startup
#
# This script is idempotent — it skips files that already exist.
# Run as: bash /workspace/scripts/download_models.sh
# =============================================================================

set -e

echo "============================================="
echo "  Multi-Model AI Platform — Model Downloader"
echo "============================================="

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure required directories exist
log_info "Ensuring directories exist..."
for dir in "${MODELS_DIR}/llm" "${MODELS_DIR}/video" "${MODELS_DIR}/audio" "${COMFYUI_DIR}/models/checkpoints" "/workspace/outputs"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    else
        log_info "Directory already exists: $dir"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 1. SDXL 1.0 Checkpoint (for ComfyUI)
# ─────────────────────────────────────────────────────────────────────────────
SDXL_PATH="${COMFYUI_DIR}/models/checkpoints/sd_xl_base_1.0.safetensors"
if [ ! -s "$SDXL_PATH" ]; then
    log_info "Downloading SDXL 1.0 base checkpoint..."
    wget -q --show-progress -O "$SDXL_PATH" \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    log_info "SDXL download complete ($(du -h "$SDXL_PATH" | cut -f1))"
else
    log_info "SDXL checkpoint already exists, skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dolphin-Mistral-Nemo 12B GGUF (for text generation)
# ─────────────────────────────────────────────────────────────────────────────
MIXTRAL_PATH="${MODELS_DIR}/llm/dolphin-2.9.4-mistral-nemo-12b-Q4_K_M.gguf"
if [ ! -s "$MIXTRAL_PATH" ]; then
    log_info "Downloading Dolphin-Mistral-Nemo 12B (Q4_K_M)..."
    wget -q --show-progress -O "$MIXTRAL_PATH" \
        "https://huggingface.co/bartowski/dolphin-2.9.4-mistral-nemo-12b-GGUF/resolve/main/dolphin-2.9.4-mistral-nemo-12b-Q4_K_M.gguf"
    log_info "Mistral-Nemo download complete ($(du -h "$MIXTRAL_PATH" | cut -f1))"
else
    log_info "Mistral-Nemo GGUF already exists, skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Wan 2.1 T2V 1.3B (download via wget)
# ─────────────────────────────────────────────────────────────────────────────
WAN_T2V_DIR="${MODELS_DIR}/video/wan2.1-t2v-1.3b"
if [ ! -s "${WAN_T2V_DIR}/transformer/diffusion_pytorch_model.safetensors" ]; then
    log_info "Downloading Wan 2.1 T2V 1.3B model files..."
    mkdir -p "${WAN_T2V_DIR}"
    WAN_T2V_BASE="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/resolve/main"

    # Download config and index files
    for f in model_index.json scheduler/scheduler_config.json; do
        dir=$(dirname "${WAN_T2V_DIR}/${f}")
        mkdir -p "$dir"
        wget -q --show-progress -O "${WAN_T2V_DIR}/${f}" "${WAN_T2V_BASE}/${f}" || true
    done

    # Download transformer weights
    mkdir -p "${WAN_T2V_DIR}/transformer"
    wget -q --show-progress -O "${WAN_T2V_DIR}/transformer/config.json" "${WAN_T2V_BASE}/transformer/config.json"
    wget -q --show-progress -O "${WAN_T2V_DIR}/transformer/diffusion_pytorch_model.safetensors" \
        "${WAN_T2V_BASE}/transformer/diffusion_pytorch_model.safetensors"

    # Download VAE
    mkdir -p "${WAN_T2V_DIR}/vae"
    wget -q --show-progress -O "${WAN_T2V_DIR}/vae/config.json" "${WAN_T2V_BASE}/vae/config.json"
    wget -q --show-progress -O "${WAN_T2V_DIR}/vae/diffusion_pytorch_model.safetensors" \
        "${WAN_T2V_BASE}/vae/diffusion_pytorch_model.safetensors"

    # Download text encoder and tokenizer
    for subdir in text_encoder tokenizer; do
        mkdir -p "${WAN_T2V_DIR}/${subdir}"
        wget -q --show-progress -O "${WAN_T2V_DIR}/${subdir}/config.json" "${WAN_T2V_BASE}/${subdir}/config.json" || true
        # Download safetensors or bin model file
        wget -q --show-progress -O "${WAN_T2V_DIR}/${subdir}/model.safetensors" \
            "${WAN_T2V_BASE}/${subdir}/model.safetensors" 2>/dev/null || \
        wget -q --show-progress -O "${WAN_T2V_DIR}/${subdir}/pytorch_model.bin" \
            "${WAN_T2V_BASE}/${subdir}/pytorch_model.bin" || true
    done

    log_info "Wan T2V download complete"
else
    log_info "Wan T2V model already exists, skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Wan 2.1 I2V 14B 480P (download via wget)
# ─────────────────────────────────────────────────────────────────────────────
WAN_I2V_DIR="${MODELS_DIR}/video/wan2.1-i2v-14b"
if [ ! -s "${WAN_I2V_DIR}/transformer/diffusion_pytorch_model-00001-of-00006.safetensors" ]; then
    log_info "Downloading Wan 2.1 I2V 14B model files..."
    mkdir -p "${WAN_I2V_DIR}"
    WAN_I2V_BASE="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/resolve/main"

    # Download config and index files
    for f in model_index.json scheduler/scheduler_config.json; do
        dir=$(dirname "${WAN_I2V_DIR}/${f}")
        mkdir -p "$dir"
        wget -q --show-progress -O "${WAN_I2V_DIR}/${f}" "${WAN_I2V_BASE}/${f}" || true
    done

    # Download transformer (sharded — multiple files)
    mkdir -p "${WAN_I2V_DIR}/transformer"
    wget -q --show-progress -O "${WAN_I2V_DIR}/transformer/config.json" "${WAN_I2V_BASE}/transformer/config.json"
    wget -q --show-progress -O "${WAN_I2V_DIR}/transformer/diffusion_pytorch_model.safetensors.index.json" \
        "${WAN_I2V_BASE}/transformer/diffusion_pytorch_model.safetensors.index.json" || true
    # Download all shards
    for i in $(seq -w 1 6); do
        SHARD="diffusion_pytorch_model-0000${i}-of-00006.safetensors"
        if [ ! -s "${WAN_I2V_DIR}/transformer/${SHARD}" ]; then
            wget -q --show-progress -O "${WAN_I2V_DIR}/transformer/${SHARD}" \
                "${WAN_I2V_BASE}/transformer/${SHARD}" || log_warn "Failed to download shard ${SHARD}"
        fi
    done

    # Download VAE
    mkdir -p "${WAN_I2V_DIR}/vae"
    wget -q --show-progress -O "${WAN_I2V_DIR}/vae/config.json" "${WAN_I2V_BASE}/vae/config.json"
    wget -q --show-progress -O "${WAN_I2V_DIR}/vae/diffusion_pytorch_model.safetensors" \
        "${WAN_I2V_BASE}/vae/diffusion_pytorch_model.safetensors"

    # Download image encoder and text encoder
    for subdir in image_encoder text_encoder tokenizer; do
        mkdir -p "${WAN_I2V_DIR}/${subdir}"
        wget -q --show-progress -O "${WAN_I2V_DIR}/${subdir}/config.json" "${WAN_I2V_BASE}/${subdir}/config.json" || true
        wget -q --show-progress -O "${WAN_I2V_DIR}/${subdir}/model.safetensors" \
            "${WAN_I2V_BASE}/${subdir}/model.safetensors" 2>/dev/null || \
        wget -q --show-progress -O "${WAN_I2V_DIR}/${subdir}/pytorch_model.bin" \
            "${WAN_I2V_BASE}/${subdir}/pytorch_model.bin" || true
    done

    log_info "Wan I2V download complete"
else
    log_info "Wan I2V model already exists, skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. XTTS v2 (auto-downloads via TTS library on first use, but we can pre-cache)
# ─────────────────────────────────────────────────────────────────────────────
log_info "Pre-caching XTTS v2 model..."
CUDA_VISIBLE_DEVICES="" python3 -c "
from TTS.api import TTS
tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)
print('XTTS v2 cached successfully')
" || log_warn "XTTS v2 pre-cache failed (will download on first request)"

echo ""
echo "============================================="
echo "  Model download complete!"
echo "============================================="
echo ""
echo "Disk usage:"
du -sh "${COMFYUI_DIR}/models/checkpoints/" 2>/dev/null || true
du -sh "${MODELS_DIR}/llm/" 2>/dev/null || true
echo ""
echo "Total models directory:"
du -sh "${MODELS_DIR}" 2>/dev/null || true
