#!/bin/bash
# Wrapper to add color to ComfyUI logs

# Green color code
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Run ComfyUI and pipe through sed to color the "Ready" message
# We use --unbuffered to ensure logs appear in real-time
python /workspace/ComfyUI/main.py "$@" 2>&1 | sed --unbuffered "s|To see the GUI go to: http://127.0.0.1:8188|${GREEN}To see the GUI go to: http://127.0.0.1:8188${NC}|g"
