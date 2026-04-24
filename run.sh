#!/bin/bash
# =============================================================================
# Cross-Platform Runner for Multi-Model AI Platform
#
# Usage:
#   ./run.sh [extra-docker-compose-args]
# =============================================================================

COMPOSE_FILES="-f docker-compose.yml"

# 1. Detect if NVIDIA GPU is present and driver is installed
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi -L &> /dev/null; then
        echo "✅ NVIDIA GPU detected. Enabling hardware acceleration..."
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.gpu.yml"
        
        # Check if MIXTRAL_GPU_LAYERS is still set to 0 and warn the user
        if grep -q "MIXTRAL_GPU_LAYERS=0" .env; then
            echo "⚠️  Note: MIXTRAL_GPU_LAYERS is set to 0 in .env. For better performance on Linux, consider setting it to -1."
        fi
    else
        echo "❌ NVIDIA GPU driver found but no devices detected. Falling back to CPU..."
    fi
else
    echo "ℹ️  No NVIDIA GPU detected (or on macOS). Using CPU mode..."
fi

# 2. Run Docker Compose
echo "🚀 Starting containers using: docker compose $COMPOSE_FILES up ${@:- --build}"
docker compose $COMPOSE_FILES up "$@"
