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
# 1. Verify ComfyUI exists (pre-installed in Dockerfile)
# ─────────────────────────────────────────────────────────────────
if [ ! -d "${COMFYUI_DIR}" ]; then
    echo "[SETUP] ComfyUI not found — cloning (fallback for volume mounts)..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_DIR}"
    cd "${COMFYUI_DIR}"
    pip install -r requirements.txt --quiet
    echo "[SETUP] ComfyUI installed"
else
    echo "[SETUP] ComfyUI found at ${COMFYUI_DIR}"
fi

# ─────────────────────────────────────────────────────────────────
# 3. Download models
# ─────────────────────────────────────────────────────────────────
echo "[SETUP] Running model download script..."
bash "${APP_ROOT}/scripts/download_models.sh"

# ─────────────────────────────────────────────────────────────────
# 4. Create log directory
# ─────────────────────────────────────────────────────────────────
mkdir -p /var/log/supervisor

# ─────────────────────────────────────────────────────────────────
# 5. Start supervisord (fires up ComfyUI + FastAPI)
# ─────────────────────────────────────────────────────────────────
echo "[SETUP] Starting supervisord..."
exec /usr/bin/supervisord -n -c "${APP_ROOT}/supervisord.conf"
