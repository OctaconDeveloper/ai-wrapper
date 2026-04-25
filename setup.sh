#!/bin/bash
# =============================================================================
# Setup Script — Container entrypoint / Vast.ai provisioning script
#
# This is the main startup script. It:
# 1. Downloads models (if not already cached)
# 2. Sets up ComfyUI
# 3. Starts supervisord (which manages ComfyUI + FastAPI)
# =============================================================================

set -e

echo "============================================================"
echo "  Multi-Model AI Platform — Container Setup"
echo "  $(date)"
echo "============================================================"

WORKSPACE="/workspace"
APP_ROOT="${WORKSPACE}/app-root"
COMFYUI_DIR="${WORKSPACE}/ComfyUI"
MODELS_DIR="${WORKSPACE}/models"

# ─────────────────────────────────────────────────────────────────
# 0. Set Environment Flags based on DEVICE
# ─────────────────────────────────────────────────────────────────
if [ "$DEVICE" = "cpu" ]; then
    echo "[SETUP] CPU mode detected. Setting flags..."
    export COMFYUI_ARGS="--cpu"
    export CUDA_VISIBLE_DEVICES=""
else
    echo "[SETUP] GPU mode detected."
    export COMFYUI_ARGS=""
fi

# ─────────────────────────────────────────────────────────────────
# 1. Verify ComfyUI exists (pre-installed in Dockerfile)
# ─────────────────────────────────────────────────────────────────
if [ ! -d "${COMFYUI_DIR}" ]; then
    echo "[SETUP] ComfyUI not found — cloning (fallback for volume mounts)..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_DIR}"
    cd "${COMFYUI_DIR}"
    python3 -m pip install -r requirements.txt --quiet
    echo "[SETUP] ComfyUI installed"
else
    echo "[SETUP] ComfyUI found at ${COMFYUI_DIR}"
fi

# ─────────────────────────────────────────────────────────────────
# 3. Download models
# ─────────────────────────────────────────────────────────────────
echo "[SETUP] Running model download script..."
bash "${APP_ROOT}/scripts/download_models.sh"
echo "[SETUP] Model download script finished with code $?"

# ─────────────────────────────────────────────────────────────────
# 4. Create log directory
# ─────────────────────────────────────────────────────────────────
mkdir -p /var/log/supervisor /workspace/comfyui_user_0 /workspace/comfyui_user_1

# ─────────────────────────────────────────────────────────────────
# 5. Start supervisord (fires up ComfyUI + FastAPI)
# ─────────────────────────────────────────────────────────────────
echo "[SETUP] Starting supervisord..."
exec /usr/bin/supervisord -n -c "${APP_ROOT}/supervisord.conf"
