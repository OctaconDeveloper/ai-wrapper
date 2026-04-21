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
        # We track pipe instances per device since we support model duplication
        self._t2v_pipes: dict[int, Any] = {}
        self._i2v_pipes: dict[int, Any] = {}

    def _load_t2v(self, device_id: int):
        """Lazy-load the text-to-video pipeline on a specific GPU."""
        state = model_manager.get_state(ModelType.VIDEO_T2V, device_id)
        if not state:
            state = model_manager.register(ModelType.VIDEO_T2V, f"Wan 2.1 T2V (GPU {device_id})", device_id)

        if state.instance is not None:
            state.touch()
            return state.instance

        logger.info(f"Loading Wan 2.1 T2V model on cuda:{device_id}: {settings.wan_t2v_model}")
        state.is_loading = True

        try:
            from diffusers import WanPipeline

            # Ensure capacity on this specific GPU
            # Note: capacity check is handled by the queue service caller
            
            pipe = WanPipeline.from_pretrained(
                settings.wan_t2v_model,
                torch_dtype=torch.bfloat16,
            )

            if settings.enable_model_offload:
                pipe.enable_model_cpu_offload(gpu_id=device_id)
            else:
                pipe.to(f"cuda:{device_id}")

            state.mark_loaded(pipe, vram_mb=6000, unload_callback=self.unload)
            logger.info(f"Wan 2.1 T2V loaded on GPU {device_id} successfully")
            return pipe

        except Exception as e:
            state.mark_error(str(e))
            logger.error(f"Failed to load Wan T2V on GPU {device_id}: {e}")
            raise

    def _load_i2v(self, device_id: int):
        """Lazy-load the image-to-video pipeline on a specific GPU."""
        state = model_manager.get_state(ModelType.VIDEO_I2V, device_id)
        if not state:
            state = model_manager.register(ModelType.VIDEO_I2V, f"Wan 2.1 I2V (GPU {device_id})", device_id)

        if state.instance is not None:
            state.touch()
            return state.instance

        logger.info(f"Loading Wan 2.1 I2V model on cuda:{device_id}: {settings.wan_i2v_model}")
        state.is_loading = True

        try:
            from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
            from transformers import CLIPVisionModel

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

            pipe = WanImageToVideoPipeline.from_pretrained(
                settings.wan_i2v_model,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=torch.bfloat16,
            )

            if settings.enable_model_offload:
                pipe.enable_model_cpu_offload(gpu_id=device_id)
            else:
                pipe.to(f"cuda:{device_id}")

            state.mark_loaded(pipe, vram_mb=28000, unload_callback=self.unload)
            logger.info(f"Wan 2.1 I2V loaded on GPU {device_id} successfully")
            return pipe

        except Exception as e:
            state.mark_error(str(e))
            logger.error(f"Failed to load Wan I2V on GPU {device_id}: {e}")
            raise

    def unload(self, device_id: int):
        """Unload models from memory for a specific GPU."""
        for mt in [ModelType.VIDEO_T2V, ModelType.VIDEO_I2V]:
            state = model_manager.get_state(mt, device_id)
            if state and state.instance is not None:
                del state.instance
                state.mark_unloaded()
                logger.info(f"Wan 2.1 {mt.value} unloaded from GPU {device_id}")

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
        device_id: int = 0, # Added device_id
    ) -> dict:
        """Generate video from text prompt."""
        start_time = time.time()

        # Capacity management and load
        await model_manager.ensure_capacity(device_id, exclude_type=ModelType.VIDEO_T2V)
        pipe = self._load_t2v(device_id)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=f"cuda:{device_id}").manual_seed(seed)

        logger.info(f"Generating T2V on GPU {device_id}: '{prompt[:80]}...'")

        # Offload inference to thread
        output = await asyncio.to_thread(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # Export frames to MP4 (also in thread)
        video_b64 = await asyncio.to_thread(self._frames_to_mp4_base64, output.frames[0], fps=fps)

        elapsed = time.time() - start_time
        logger.info(f"T2V on GPU {device_id} complete in {elapsed:.1f}s")

        return {
            "video_base64": video_b64,
            "prompt": prompt,
            "mode": "t2v",
            "num_frames": num_frames,
            "generation_time_seconds": round(elapsed, 2),
            "device_id": device_id,
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
        device_id: int = 0, # Added device_id
    ) -> dict:
        """Generate video from image + text prompt."""
        start_time = time.time()
        from PIL import Image as PILImage

        # Capacity management and load
        await model_manager.ensure_capacity(device_id, exclude_type=ModelType.VIDEO_I2V)
        pipe = self._load_i2v(device_id)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        # Decode input image
        img_bytes = base64.b64decode(image_base64)
        input_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        input_image = input_image.resize((width, height))

        generator = torch.Generator(device=f"cuda:{device_id}").manual_seed(seed)

        logger.info(f"Generating I2V on GPU {device_id}: '{prompt[:80]}...'")

        # Offload inference to thread
        output = await asyncio.to_thread(
            pipe,
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

        video_b64 = await asyncio.to_thread(self._frames_to_mp4_base64, output.frames[0], fps=fps)

        elapsed = time.time() - start_time
        logger.info(f"I2V on GPU {device_id} complete in {elapsed:.1f}s")

        return {
            "video_base64": video_b64,
            "prompt": prompt,
            "mode": "i2v",
            "num_frames": num_frames,
            "generation_time_seconds": round(elapsed, 2),
            "device_id": device_id,
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
