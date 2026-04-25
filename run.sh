#!/bin/bash
# =============================================================================
# Cross-Platform Runner for Multi-Model AI Platform
#
# Usage:
#   ./run.sh [extra-docker-compose-args]
# =============================================================================

COMPOSE_FILES="-f docker-compose.yml"

# 1. Detect environment
DEVICE="cuda"
BASE_IMAGE="nvidia/cuda:12.4.1-devel-ubuntu22.04"
GPU_LAYERS="-1"

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected. Using CPU mode (lightweight base image)."
    DEVICE="cpu"
    BASE_IMAGE="ubuntu:22.04"
    GPU_LAYERS="0"
    echo "⚠️  IMPORTANT: Dolphin-Mixtral (25GB) requires significant RAM."
    echo "   Ensure Docker Desktop has at least 32GB of RAM allocated."
    echo "   (Settings -> Resources -> Memory)"
elif ! command -v nvidia-smi &> /dev/null || ! nvidia-smi -L &> /dev/null; then
    echo "ℹ️  No NVIDIA GPU detected. Using CPU mode (lightweight base image)..."
    DEVICE="cpu"
    BASE_IMAGE="ubuntu:22.04"
    GPU_LAYERS="0"
fi

if [ "$DEVICE" = "cuda" ]; then
    echo "✅ NVIDIA GPU detected. Enabling hardware acceleration..."
    COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.gpu.yml"
    
    # Check if MIXTRAL_GPU_LAYERS is still set to 0 and warn the user
    if grep -q "MIXTRAL_GPU_LAYERS=0" .env; then
        echo "⚠️  Note: MIXTRAL_GPU_LAYERS is set to 0 in .env. For better performance on Linux, consider setting it to -1."
    fi
fi

# 2. Run Docker Compose
echo "🚀 Starting containers (Device: $DEVICE)"
COMPOSE_BAKE=true DEVICE=$DEVICE BASE_IMAGE=$BASE_IMAGE MIXTRAL_GPU_LAYERS=$GPU_LAYERS docker compose $COMPOSE_FILES up --build "$@"
