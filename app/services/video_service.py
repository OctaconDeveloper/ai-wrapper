"""
Video Service — Wan 2.1 text-to-video and image-to-video generation.

Uses HuggingFace diffusers pipelines with lazy loading.
T2V uses the lightweight 1.3B model, I2V uses the 14B 480P model.
"""

from __future__ import annotations

import base64
import io
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from app.config import settings
from app.services.model_manager import ModelManager, ModelType, model_manager

logger = logging.getLogger(__name__)


class VideoService:
    """Handles text-to-video and image-to-video via Wan 2.1."""

    def __init__(self):
        self._t2v_pipe = None
        self._i2v_pipe = None
        self._t2v_state = model_manager.register(ModelType.VIDEO_T2V, "Wan 2.1 T2V 1.3B")
        self._i2v_state = model_manager.register(ModelType.VIDEO_I2V, "Wan 2.1 I2V 14B")

    def _load_t2v(self):
        """Lazy-load the text-to-video pipeline."""
        if self._t2v_pipe is not None:
            self._t2v_state.touch()
            return

        logger.info(f"Loading Wan 2.1 T2V model: {settings.wan_t2v_model}")
        self._t2v_state.is_loading = True

        try:
            from diffusers import WanPipeline

            # Check if we need to evict another model
            if model_manager.should_evict():
                candidate = model_manager.get_eviction_candidate(exclude=ModelType.VIDEO_T2V)
                if candidate:
                    self._evict_model(candidate)

            self._t2v_pipe = WanPipeline.from_pretrained(
                settings.wan_t2v_model,
                torch_dtype=torch.bfloat16,
            )

            if settings.enable_model_offload:
                self._t2v_pipe.enable_model_cpu_offload()
            else:
                self._t2v_pipe.to("cuda")

            self._t2v_state.mark_loaded(self._t2v_pipe, vram_mb=6000)
            logger.info("Wan 2.1 T2V loaded successfully")

        except Exception as e:
            self._t2v_state.mark_error(str(e))
            logger.error(f"Failed to load Wan T2V: {e}")
            raise

    def _load_i2v(self):
        """Lazy-load the image-to-video pipeline."""
        if self._i2v_pipe is not None:
            self._i2v_state.touch()
            return

        logger.info(f"Loading Wan 2.1 I2V model: {settings.wan_i2v_model}")
        self._i2v_state.is_loading = True

        try:
            from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
            from transformers import CLIPVisionModel

            if model_manager.should_evict():
                candidate = model_manager.get_eviction_candidate(exclude=ModelType.VIDEO_I2V)
                if candidate:
                    self._evict_model(candidate)

            # Load components
            image_encoder = CLIPVisionModel.from_pretrained(
                settings.wan_i2v_model,
                subfolder="image_encoder",
                torch_dtype=torch.float32,
            )
            vae = AutoencoderKLWan.from_pretrained(
                settings.wan_i2v_model,
                subfolder="vae",
                torch_dtype=torch.float32,
            )

            self._i2v_pipe = WanImageToVideoPipeline.from_pretrained(
                settings.wan_i2v_model,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=torch.bfloat16,
            )

            if settings.enable_model_offload:
                self._i2v_pipe.enable_model_cpu_offload()
            else:
                self._i2v_pipe.to("cuda")

            self._i2v_state.mark_loaded(self._i2v_pipe, vram_mb=28000)
            logger.info("Wan 2.1 I2V loaded successfully")

        except Exception as e:
            self._i2v_state.mark_error(str(e))
            logger.error(f"Failed to load Wan I2V: {e}")
            raise

    def _evict_model(self, model_type: ModelType):
        """Unload a model to free VRAM."""
        logger.info(f"Evicting model: {model_type.value}")
        if model_type == ModelType.VIDEO_T2V and self._t2v_pipe is not None:
            del self._t2v_pipe
            self._t2v_pipe = None
            self._t2v_state.mark_unloaded()
        elif model_type == ModelType.VIDEO_I2V and self._i2v_pipe is not None:
            del self._i2v_pipe
            self._i2v_pipe = None
            self._i2v_state.mark_unloaded()

        ModelManager.clear_gpu_cache()

    async def generate_t2v(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted",
        num_frames: int = 33,
        width: int = 480,
        height: int = 320,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        seed: int = -1,
        fps: int = 16,
    ) -> dict:
        """Generate video from text prompt."""
        start_time = time.time()

        # Lazy load
        self._load_t2v()

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(f"Generating T2V: '{prompt[:80]}...' ({num_frames} frames, {width}x{height})")

        output = self._t2v_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # Export frames to MP4
        video_b64 = self._frames_to_mp4_base64(output.frames[0], fps=fps)

        elapsed = time.time() - start_time
        logger.info(f"T2V complete in {elapsed:.1f}s")

        return {
            "video_base64": video_b64,
            "prompt": prompt,
            "mode": "t2v",
            "num_frames": num_frames,
            "generation_time_seconds": round(elapsed, 2),
        }

    async def generate_i2v(
        self,
        prompt: str,
        image_base64: str,
        negative_prompt: str = "blurry, low quality, distorted",
        num_frames: int = 33,
        width: int = 480,
        height: int = 320,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        seed: int = -1,
        fps: int = 16,
    ) -> dict:
        """Generate video from image + text prompt."""
        start_time = time.time()
        from PIL import Image as PILImage

        # Lazy load
        self._load_i2v()

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        # Decode input image
        img_bytes = base64.b64decode(image_base64)
        input_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        input_image = input_image.resize((width, height))

        generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(f"Generating I2V: '{prompt[:80]}...' ({num_frames} frames)")

        output = self._i2v_pipe(
            prompt=prompt,
            image=input_image,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        video_b64 = self._frames_to_mp4_base64(output.frames[0], fps=fps)

        elapsed = time.time() - start_time
        logger.info(f"I2V complete in {elapsed:.1f}s")

        return {
            "video_base64": video_b64,
            "prompt": prompt,
            "mode": "i2v",
            "num_frames": num_frames,
            "generation_time_seconds": round(elapsed, 2),
        }

    @staticmethod
    def _frames_to_mp4_base64(frames, fps: int = 16) -> str:
        """Convert a list of PIL images / numpy frames to a base64-encoded MP4."""
        import imageio

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        # Convert frames to numpy arrays if needed
        np_frames = []
        for frame in frames:
            if hasattr(frame, "numpy"):
                # Tensor
                arr = frame.cpu().numpy()
            elif hasattr(frame, "convert"):
                # PIL Image
                arr = np.array(frame)
            else:
                arr = np.array(frame)

            # Ensure uint8
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)

            np_frames.append(arr)

        # Write MP4
        writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()

        # Read and encode
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()

        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

        return base64.b64encode(video_bytes).decode("utf-8")


# Global singleton
video_service = VideoService()
