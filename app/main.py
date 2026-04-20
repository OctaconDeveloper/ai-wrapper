"""
Multi-Model AI Platform — FastAPI Gateway

Single unified API for:
  - POST /api/image/generate    (SDXL via ComfyUI)
  - POST /api/video/generate    (Wan 2.1 T2V / I2V)
  - POST /api/text/generate     (Dolphin-Mixtral 8x7B)
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
    logger.info(f"  ComfyUI: {settings.comfyui_url}")
    logger.info(f"  Models:  {settings.models_dir}")
    logger.info("=" * 60)

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info(f"  GPU:     {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("  GPU:     No CUDA GPU detected! Models will be slow.")
    except ImportError:
        logger.warning("  GPU:     PyTorch not available")

    # Start idle shutdown monitor
    await idle_shutdown_service.start()

    yield

    # Shutdown
    logger.info("Shutting down...")
    await idle_shutdown_service.stop()
    await image_service.close()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Multi-Model AI Platform",
    description=(
        "Unified API for uncensored AI generation: "
        "images (SDXL), video (Wan 2.1), text (Dolphin-Mixtral), audio (XTTS v2)"
    ),
    version="1.0.0",
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
# Activity Tracking Middleware (resets idle shutdown timer)
# =============================================================================

# Paths that do NOT count as "activity" (don't reset idle timer)
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

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health and model status."""
    gpu_info = model_manager.get_gpu_memory_info()
    model_states = model_manager.get_all_states()

    # Check ComfyUI
    comfyui_ok = await image_service.health_check()
    model_states["comfyui"] = {
        "name": "ComfyUI (SDXL)",
        "status": "loaded" if comfyui_ok else "unloaded",
        "vram_mb": 7000 if comfyui_ok else 0,
    }

    return HealthResponse(
        status="ok",
        models={k: v["status"] for k, v in model_states.items()},
        gpu_memory_used_mb=gpu_info.get("used_mb", 0),
        gpu_memory_total_mb=gpu_info.get("total_mb", 0),
    )


@app.get("/api/models", tags=["System"])
async def list_models():
    """List all available models and their current state."""
    return {
        "models": model_manager.get_all_states(),
        "gpu": model_manager.get_gpu_memory_info(),
        "config": {
            "max_loaded_models": settings.max_loaded_models,
            "model_offload_enabled": settings.enable_model_offload,
        },
    }


@app.get("/api/idle-status", tags=["System"])
async def idle_status():
    """Check idle timer and auto-shutdown countdown."""
    return idle_shutdown_service.get_status()


@app.post("/api/shutdown", tags=["System"])
async def manual_shutdown():
    """Immediately stop the vast.ai instance to save costs."""
    import asyncio
    # Respond first, then shutdown
    asyncio.get_event_loop().call_later(2.0, lambda: asyncio.ensure_future(idle_shutdown_service.force_shutdown()))
    return {"status": "shutdown_initiated", "message": "Instance will stop in ~2 seconds"}


# =============================================================================
# Image Generation — SDXL via ComfyUI
# =============================================================================

@app.post(
    "/api/image/generate",
    response_model=ImageGenerateResponse,
    tags=["Image"],
    responses={500: {"model": ErrorResponse}},
)
async def generate_image(request: ImageGenerateRequest):
    """Generate images using SDXL via ComfyUI."""
    try:
        result = await image_service.generate(
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

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
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
    responses={500: {"model": ErrorResponse}},
)
async def generate_video(request: VideoGenerateRequest):
    """Generate video using Wan 2.1 (text-to-video or image-to-video)."""
    try:
        if request.mode == VideoMode.IMAGE_TO_VIDEO:
            if not request.image_base64:
                raise HTTPException(
                    status_code=400,
                    detail="image_base64 is required for i2v mode",
                )
            result = await video_service.generate_i2v(
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
            result = await video_service.generate_t2v(
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
# Text Generation — Dolphin-Mixtral 8x7B
# =============================================================================

@app.post(
    "/api/text/generate",
    response_model=TextGenerateResponse,
    tags=["Text"],
    responses={500: {"model": ErrorResponse}},
)
async def generate_text(request: TextGenerateRequest):
    """Generate text using Dolphin-Mixtral 8x7B (uncensored)."""
    try:
        # Validate input
        if not request.prompt and not request.messages:
            raise HTTPException(
                status_code=400,
                detail="Either 'prompt' or 'messages' must be provided",
            )

        messages_dicts = None
        if request.messages:
            messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

        # Handle streaming
        if request.stream:
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
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming
        result = await text_service.generate(
            prompt=request.prompt,
            messages=messages_dicts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            seed=request.seed,
            stream=False,
        )
        return TextGenerateResponse(**result)

    except Exception as e:
        logger.error(f"Text generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Audio Generation — XTTS v2
# =============================================================================

@app.post(
    "/api/audio/generate",
    response_model=AudioGenerateResponse,
    tags=["Audio"],
    responses={500: {"model": ErrorResponse}},
)
async def generate_audio(request: AudioGenerateRequest):
    """Generate speech using XTTS v2 (with optional voice cloning)."""
    try:
        result = await audio_service.generate(
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

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,  # Single worker — models are not fork-safe
        log_level="info",
    )
