"""
Multi-Model AI Platform — FastAPI Gateway

Single unified API for:
  - POST /api/image/generate    (SDXL via ComfyUI)
  - POST /api/video/generate    (Wan 2.1 T2V / I2V)
  - POST /api/text/generate     (Dolphin-Mixtral 8x7B)
  - POST /api/text/lstm/generate (Lightweight LSTM)
  - POST /api/audio/generate    (XTTS v2)
  - GET  /api/health            (System health + model status)
  - GET  /api/models            (Available models + VRAM)
  - POST /api/shutdown           (Manual shutdown)
  - GET  /api/idle-status        (Idle timer info)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from starlette.requests import Request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import settings
from app.schemas import (
    AudioGenerateRequest,
    AudioGenerateResponse,
    ErrorResponse,
    HealthResponse,
    ImageGenerateRequest,
    ImageGenerateResponse,
    TextGenerateRequest,
    TextGenerateResponse,
    VideoGenerateRequest,
    VideoGenerateResponse,
    VideoMode,
)
from app.services.audio_service import audio_service
from app.services.image_service import image_service
from app.services.idle_shutdown import idle_shutdown_service
from app.services.model_manager import model_manager
from app.services.text_service import text_service
from app.services.video_service import video_service
from app.services.lstm_service import lstm_service
from app.services.queue_service import queue_service, Priority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# App Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("=" * 60)
    logger.info("Multi-Model AI Platform starting up...")
    logger.info(f"  API:     http://{settings.api_host}:{settings.api_port}")
    logger.info(f"  GPUs:    {model_manager.device_count}")
    logger.info("=" * 60)

    # Start services
    await idle_shutdown_service.start()
    await queue_service.start()

    yield

    # Shutdown
    logger.info("Shutting down...")
    await idle_shutdown_service.stop()
    await queue_service.stop()
    await image_service.close()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Multi-Model AI Platform",
    description=(
        "Unified Multi-GPU API for images (SDXL), video (Wan 2.1), text (Mixtral), audio (XTTS v2)"
    ),
    version="0.0.1",
    lifespan=lifespan,
)

# CORS — allow all origins for API usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Authentication Middleware (POST requests only)
# =============================================================================

@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    """Verify x-m-token header for all POST requests."""
    if request.method == "POST":
        token = request.headers.get("x-m-token")
        
        # If tokens are configured, check the header
        if settings.api_tokens:
            if not token or token not in settings.api_tokens:
                logger.warning(f"Unauthorized POST attempt to {request.url.path} from {request.client.host}")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Unauthorized", "detail": "Invalid or missing x-m-token header"}
                )
        else:
            # Optional: Log a warning if no tokens are configured but auth is expected
            # logger.warning("API_TOKENS is empty. POST requests are currently unprotected.")
            pass

    response = await call_next(request)
    return response


# =============================================================================
# Activity Tracking Middleware (resets idle shutdown timer)
# =============================================================================

_EXCLUDED_PATHS = {"/api/health", "/api/models", "/api/idle-status", "/docs", "/openapi.json"}


@app.middleware("http")
async def track_activity(request: Request, call_next):
    """Reset idle shutdown timer on real API usage."""
    if request.url.path not in _EXCLUDED_PATHS:
        idle_shutdown_service.touch()
    response = await call_next(request)
    return response


# =============================================================================
# Health & System Routes
# =============================================================================

@app.get("/api/health", tags=["System"])
async def health_check():
    """Check system health and status across all GPUs."""
    gpu_stats = [model_manager.get_gpu_memory_info(i) for i in range(model_manager.device_count)]
    model_states = model_manager.get_all_states()

    # Check all ComfyUI instances
    comfy_status = []
    for i in range(model_manager.device_count):
        ok = await image_service.health_check(i)
        comfy_status.append({"gpu": i, "status": "ok" if ok else "error"})

    return {
        "status": "ok",
        "gpu_count": model_manager.device_count,
        "gpus": gpu_stats,
        "models": model_states,
        "comfyui": comfy_status,
    }


@app.get("/api/models", tags=["System"])
async def list_models():
    """List all available models and their current state."""
    return {
        "models": model_manager.get_all_states(),
        "gpu_count": model_manager.device_count,
        "config": {
            "max_loaded_per_gpu": model_manager.max_loaded_per_gpu,
        },
    }


@app.get("/api/idle-status", tags=["System"])
async def idle_status():
    """Check idle timer and auto-shutdown countdown."""
    return idle_shutdown_service.get_status()


@app.post("/api/shutdown", tags=["System"])
async def manual_shutdown():
    """Immediately stop the instance."""
    import asyncio
    asyncio.get_event_loop().call_later(2.0, lambda: asyncio.ensure_future(idle_shutdown_service.force_shutdown()))
    return {"status": "shutdown_initiated", "message": "Instance will stop in ~2 seconds"}


# =============================================================================
# Image Generation — SDXL via Dual ComfyUI
# =============================================================================

@app.post(
    "/api/image/generate",
    response_model=ImageGenerateResponse,
    tags=["Image"],
)
async def generate_image(request: ImageGenerateRequest):
    """Generate images using prioritized queue across available GPUs."""
    try:
        result = await queue_service.enqueue(
            ModelType.IMAGE,
            Priority.IMAGE,
            image_service.generate,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            batch_size=request.batch_size,
        )
        return ImageGenerateResponse(**result)

    except Exception as e:
        logger.error(f"Image generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Video Generation — Wan 2.1
# =============================================================================

@app.post(
    "/api/video/generate",
    response_model=VideoGenerateResponse,
    tags=["Video"],
)
async def generate_video(request: VideoGenerateRequest):
    """Generate video using prioritized queue."""
    try:
        if request.mode == VideoMode.IMAGE_TO_VIDEO:
            if not request.image_base64:
                raise HTTPException(status_code=400, detail="image_base64 is required")
            
            result = await queue_service.enqueue(
                ModelType.VIDEO_I2V,
                Priority.VIDEO,
                video_service.generate_i2v,
                prompt=request.prompt,
                image_base64=request.image_base64,
                negative_prompt=request.negative_prompt,
                num_frames=request.num_frames,
                width=request.width,
                height=request.height,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                fps=request.fps,
            )
        else:
            result = await queue_service.enqueue(
                ModelType.VIDEO_T2V,
                Priority.VIDEO,
                video_service.generate_t2v,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_frames=request.num_frames,
                width=request.width,
                height=request.height,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                fps=request.fps,
            )

        return VideoGenerateResponse(**result)

    except Exception as e:
        logger.error(f"Video generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Text Generation — Mixtral
# =============================================================================

@app.post(
    "/api/text/generate",
    response_model=TextGenerateResponse,
    tags=["Text"],
)
async def generate_text(request: TextGenerateRequest):
    """Generate text with High Priority."""
    try:
        if not request.prompt and not request.messages:
            raise HTTPException(status_code=400, detail="'prompt' or 'messages' required")

        messages_dicts = None
        if request.messages:
            messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

        # Note: Streaming is technically doable via the queue but requires more complex generator handling.
        # For now, we handle non-streaming via the priority queue.
        if request.stream:
            # We bypass the queue for streaming to avoid holding up a slot during long streams,
            # or we could implement a separate "stream" slot. 
            # Simplified: Use GPU 0 for streams for now.
            logger.warning("Streaming bypasses the priority queue and uses default GPU")
            generator = await text_service.generate(
                prompt=request.prompt,
                messages=messages_dicts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repeat_penalty=request.repeat_penalty,
                seed=request.seed,
                stream=True,
                device_id=0
            )
            return StreamingResponse(generator, media_type="text/event-stream")

        result = await queue_service.enqueue(
            ModelType.TEXT,
            Priority.TEXT,
            text_service.generate,
            prompt=request.prompt,
            messages=messages_dicts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            seed=seed if (seed := request.seed) else -1
        )
        return TextGenerateResponse(**result)

    except Exception as e:
        logger.error(f"Text generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Text Generation — Lightweight LSTM
# =============================================================================

@app.post(
    "/api/text/lstm/generate",
    response_model=TextGenerateResponse,
    tags=["Text"],
)
async def generate_text_lstm(request: TextGenerateRequest):
    """Generate text using the ultra-lightweight LSTM (Highest Priority)."""
    try:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="'prompt' is required for LSTM mode")

        result = await queue_service.enqueue(
            ModelType.LSTM,
            Priority.TEXT,
            lstm_service.generate,
            prompt=request.prompt,
            max_tokens=request.max_tokens if request.max_tokens < 500 else 100,
            temperature=request.temperature,
        )
        return TextGenerateResponse(**result)

    except Exception as e:
        logger.error(f"LSTM generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Audio Generation — XTTS v2
# =============================================================================

@app.post(
    "/api/audio/generate",
    response_model=AudioGenerateResponse,
    tags=["Audio"],
)
async def generate_audio(request: AudioGenerateRequest):
    """Generate speech using prioritized queue."""
    try:
        result = await queue_service.enqueue(
            ModelType.AUDIO,
            Priority.AUDIO,
            audio_service.generate,
            text=request.text,
            language=request.language,
            speaker_wav_base64=request.speaker_wav_base64,
            speed=request.speed,
        )
        return AudioGenerateResponse(**result)

    except Exception as e:
        logger.error(f"Audio generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, workers=1)
