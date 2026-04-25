#!/bin/bash
# =============================================================================
# Turbo Model Downloader — Parallel & High-Speed
# =============================================================================

set -e

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"
HF_HUB_ENABLE_HF_TRANSFER=1

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_turbo() { echo -e "${CYAN}[TURBO]${NC} $1"; }

# ─────────────────────────────────────────────────────────────────
# Helper: Multi-threaded Download
# ─────────────────────────────────────────────────────────────────
download_file() {
    local url=$1
    local dest=$2
    local filename=$(basename "$dest")
    local dir=$(dirname "$dest")

    mkdir -p "$dir"

    if [ -s "$dest" ]; then
        log_info "$filename already exists, skipping."
        return 0
    fi

    log_turbo "Starting download: $filename"
    if command -v aria2c &> /dev/null; then
        aria2c -x 16 -s 16 -k 1M --continue=true --console-log-level=error --summary-interval=0 --dir="$dir" --out="$filename" "$url"
    else
        wget -q --continue --show-progress -O "$dest" "$url"
    fi
}

# ─────────────────────────────────────────────────────────────────
# Download Tasks
# ─────────────────────────────────────────────────────────────────

# 1. SD 1.5 (Image - DreamShaper 8)
task_image() {
    download_file "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors" \
                  "${COMFYUI_DIR}/models/checkpoints/DreamShaper_8.safetensors"
}

# 2. Llama 3.2 3B Abliterated (Text)
task_text() {
    download_file "https://huggingface.co/MaziyarPanahi/Llama-3.2-3B-Instruct-Abliterated-GGUF/resolve/main/Llama-3.2-3B-Instruct-Abliterated.Q4_K_M.gguf" \
                  "${MODELS_DIR}/llm/Llama-3.2-3B-Instruct-Abliterated.Q4_K_M.gguf"
}

# 3. XTTS v2 (Audio)
task_audio() {
    log_turbo "Pre-caching XTTS v2..."
    CUDA_VISIBLE_DEVICES="" python3 -c "from TTS.api import TTS; TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)" &> /dev/null
    log_info "Audio model ready."
}

# 4. Wan 2.1 (Video) - The Big One
task_video() {
    if [ "$SKIP_VIDEO" = "true" ]; then
        log_warn "SKIP_VIDEO=true. Skipping 30GB video models."
        return 0
    fi

    log_turbo "Starting Background Video Download (30GB)..."
    
    # Use huggingface-cli for the sharded video models (much faster)
    # T2V 1.3B
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "${MODELS_DIR}/video/wan2.1-t2v-1.3b" --quiet &
    
    # I2V 14B (The 25GB monster)
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --local-dir "${MODELS_DIR}/video/wan2.1-i2v-14b" --quiet &
    
    wait
    log_info "Background Video downloads finished."
}

# ─────────────────────────────────────────────────────────────────
# Execution Strategy
# ─────────────────────────────────────────────────────────────────

log_turbo "Accelerating startup: Prioritizing Text and Image models..."

# Start Text and Image in parallel
task_text &
pid_text=$!

task_image &
pid_image=$!

# Audio is usually fast, run it too
task_audio &
pid_audio=$!

log_info "Waiting for essential models (Text/Image/Audio) to complete..."
wait $pid_text $pid_image $pid_audio

log_info "✅ Essential models ready! Launching API services..."

# Now start Video in the background and DO NOT wait for it
if [ "$SKIP_VIDEO" != "true" ]; then
    log_warn "Video models will continue downloading in the background."
    # Run in background with nohup to survive shell exit
    nohup bash -c "export MODELS_DIR=$MODELS_DIR; $(declare -f download_file); $(declare -f task_video); task_video" > /workspace/video_download.log 2>&1 &
fi

echo "============================================="
echo "  Core Platform Ready! "
echo "============================================="
