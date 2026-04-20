"""
Pydantic schemas for all API request/response models.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Common
# =============================================================================

class ModelStatus(str, Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"
    LOADING = "loading"
    ERROR = "error"


class HealthResponse(BaseModel):
    status: str = "ok"
    models: dict[str, str] = {}
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# =============================================================================
# Image Generation (SDXL via ComfyUI)
# =============================================================================

class ImageGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = Field(
        default="blurry, low quality, distorted, deformed",
        description="Negative prompt to avoid unwanted features",
    )
    width: int = Field(default=1024, ge=512, le=2048, description="Image width")
    height: int = Field(default=1024, ge=512, le=2048, description="Image height")
    steps: int = Field(default=25, ge=1, le=100, description="Number of diffusion steps")
    cfg_scale: float = Field(default=7.0, ge=1.0, le=20.0, description="CFG guidance scale")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    batch_size: int = Field(default=1, ge=1, le=4, description="Number of images to generate")


class ImageGenerateResponse(BaseModel):
    images: list[str] = Field(description="Base64-encoded PNG images")
    seed: int
    prompt: str
    generation_time_seconds: float


# =============================================================================
# Video Generation (Wan 2.1)
# =============================================================================

class VideoMode(str, Enum):
    TEXT_TO_VIDEO = "t2v"
    IMAGE_TO_VIDEO = "i2v"


class VideoGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    mode: VideoMode = Field(default=VideoMode.TEXT_TO_VIDEO, description="t2v or i2v")
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded input image (required for i2v mode)",
    )
    negative_prompt: str = Field(
        default="blurry, low quality, distorted",
        description="Negative prompt",
    )
    num_frames: int = Field(default=33, ge=8, le=81, description="Number of frames")
    width: int = Field(default=480, ge=256, le=1280, description="Video width")
    height: int = Field(default=320, ge=256, le=720, description="Video height")
    guidance_scale: float = Field(default=5.0, ge=1.0, le=15.0)
    num_inference_steps: int = Field(default=30, ge=10, le=100)
    seed: int = Field(default=-1)
    fps: int = Field(default=16, ge=8, le=30, description="Output FPS")


class VideoGenerateResponse(BaseModel):
    video_base64: str = Field(description="Base64-encoded MP4 video")
    prompt: str
    mode: VideoMode
    num_frames: int
    generation_time_seconds: float


# =============================================================================
# Text Generation (Mixtral / Dolphin)
# =============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class TextGenerateRequest(BaseModel):
    prompt: Optional[str] = Field(
        default=None,
        description="Text prompt for completion mode",
    )
    messages: Optional[list[ChatMessage]] = Field(
        default=None,
        description="Chat messages for chat completion mode",
    )
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=100)
    repeat_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = Field(default=False, description="Stream response via SSE")
    seed: int = Field(default=-1)


class TextGenerateResponse(BaseModel):
    text: str = Field(description="Generated text")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time_seconds: float


# =============================================================================
# Audio Generation (XTTS v2)
# =============================================================================

class AudioGenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech")
    language: str = Field(
        default="en",
        description="Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, hi)",
    )
    speaker_wav_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded WAV file for voice cloning (optional)",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


class AudioGenerateResponse(BaseModel):
    audio_base64: str = Field(description="Base64-encoded WAV audio")
    text: str
    language: str
    duration_seconds: float
    generation_time_seconds: float
