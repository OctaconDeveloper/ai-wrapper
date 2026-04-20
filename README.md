# 🚀 Multi-Model AI Platform

Unified API for uncensored AI generation — images, video, text, and audio — deployed on **vast.ai** via a single Docker container.

## Models

| Endpoint | Model | Type | VRAM |
|:---|:---|:---|:---|
| `/api/image/generate` | **SDXL 1.0** (via ComfyUI) | Text → Image | ~7 GB |
| `/api/video/generate` | **Wan 2.1 T2V 1.3B** | Text → Video | ~6 GB |
| `/api/video/generate` | **Wan 2.1 I2V 14B** | Image → Video | ~28 GB |
| `/api/text/generate` | **Dolphin-Mixtral 8x7B** (GGUF Q4_K_M) | Text → Text | ~26 GB |
| `/api/audio/generate` | **XTTS v2** | Text → Audio | ~2 GB |

> **GPU Requirement**: A100 80GB or H100 80GB recommended.

---

## Quick Start

### Local (Docker Compose)

```bash
# 1. Set your HuggingFace token
export HF_TOKEN=hf_your_token_here

# 2. Build and run
docker compose up --build

# 3. Wait for model downloads (first run takes ~30 min)
# 4. API is live at http://localhost:8000
```

### Vast.ai Deployment

```bash
# 1. Build and push Docker image
docker build -t yourusername/ai-multi-models:latest .
docker push yourusername/ai-multi-models:latest

# 2. On vast.ai:
#    - Go to Templates → Create Template
#    - Image: yourusername/ai-multi-models:latest
#    - Launch Mode: Docker ENTRYPOINT
#    - Ports: 8000
#    - Disk Space: 200 GB (for model weights)
#    - Environment: HF_TOKEN=hf_your_token
#
# 3. Search for A100 80GB instances → Rent
# 4. Access via: https://<instance-id>.vast.ai:8000
```

---

## API Reference

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Image Generation (SDXL)
```bash
curl -X POST http://localhost:8000/api/image/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cyberpunk cityscape at night, neon lights, 8k, masterpiece",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "cfg_scale": 7.0
  }'
```

### Video Generation (Wan 2.1)
```bash
# Text-to-Video
curl -X POST http://localhost:8000/api/video/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat playing with a ball in slow motion",
    "mode": "t2v",
    "num_frames": 33,
    "width": 480,
    "height": 320
  }'

# Image-to-Video
curl -X POST http://localhost:8000/api/video/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "the scene slowly comes to life, camera pans right",
    "mode": "i2v",
    "image_base64": "<base64-encoded-image>",
    "num_frames": 33
  }'
```

### Text Generation (Dolphin-Mixtral)
```bash
# Chat mode
curl -X POST http://localhost:8000/api/text/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 1024,
    "temperature": 0.7
  }'

# Streaming (SSE)
curl -X POST http://localhost:8000/api/text/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a poem about AI."}],
    "stream": true
  }'
```

### Audio Generation (XTTS v2)
```bash
# Basic TTS
curl -X POST http://localhost:8000/api/audio/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to the multi-model AI platform.",
    "language": "en",
    "speed": 1.0
  }'

# With voice cloning
curl -X POST http://localhost:8000/api/audio/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a cloned voice speaking.",
    "language": "en",
    "speaker_wav_base64": "<base64-encoded-wav>"
  }'
```

---

## Architecture

```
┌────────────────────────────────────────────────┐
│              Docker Container                   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │          supervisord (PID 1)             │   │
│  │                                         │   │
│  │  ┌──────────────┐  ┌────────────────┐  │   │
│  │  │   ComfyUI     │  │    FastAPI      │  │   │
│  │  │  :8188 (int)  │  │   :8000 (ext)   │  │   │
│  │  │              │  │                │  │   │
│  │  │  SDXL 1.0    │  │  • /api/image  │──┤   │
│  │  │              │  │  • /api/video  │  │   │
│  │  └──────────────┘  │  • /api/text   │  │   │
│  │                     │  • /api/audio  │  │   │
│  │                     │  • /api/health │  │   │
│  │                     └────────────────┘  │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  Models (lazy-loaded with LRU eviction):       │
│  ├── SDXL 1.0 ──────────── via ComfyUI         │
│  ├── Wan 2.1 T2V 1.3B ──── via diffusers       │
│  ├── Wan 2.1 I2V 14B ───── via diffusers       │
│  ├── Dolphin-Mixtral ────── via llama-cpp       │
│  └── XTTS v2 ───────────── via TTS library     │
└────────────────────────────────────────────────┘
```

---

## Environment Variables

| Variable | Default | Description |
|:---|:---|:---|
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `API_PORT` | `8000` | FastAPI port |
| `MIXTRAL_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `MIXTRAL_CONTEXT_LENGTH` | `32768` | Context window size |
| `MAX_LOADED_MODELS` | `3` | Max concurrent models in VRAM |
| `ENABLE_MODEL_OFFLOAD` | `true` | CPU offload for video models |

See `.env.example` for all variables.

---

## Project Structure

```
ai-multi-models/
├── Dockerfile                  # Production container
├── docker-compose.yml          # Local development
├── requirements.txt            # Python dependencies
├── setup.sh                    # Container entrypoint
├── supervisord.conf            # Process manager
├── .env.example                # Environment template
├── app/
│   ├── main.py                 # FastAPI gateway (all routes)
│   ├── config.py               # Settings (env vars)
│   ├── schemas.py              # Request/response models
│   └── services/
│       ├── model_manager.py    # Lazy loader + VRAM manager
│       ├── image_service.py    # SDXL via ComfyUI API
│       ├── video_service.py    # Wan 2.1 T2V + I2V
│       ├── text_service.py     # Dolphin-Mixtral (GGUF)
│       └── audio_service.py    # XTTS v2 TTS
├── comfyui/
│   └── workflows/
│       └── sdxl_txt2img.json   # ComfyUI API workflow
└── scripts/
    └── download_models.sh      # Model weight downloader
```

## License

For personal/research use. Individual model licenses apply:
- SDXL: CreativeML Open RAIL++-M
- Wan 2.1: Apache 2.0
- Dolphin-Mixtral: Apache 2.0
- XTTS v2: Coqui Public Model License
