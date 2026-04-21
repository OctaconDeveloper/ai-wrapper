"""
Image Service — SDXL generation via ComfyUI's API.

ComfyUI runs as a separate process (managed by supervisord) and exposes
a WebSocket + REST API. This service sends workflow JSON to ComfyUI,
waits for completion, and retrieves the generated images.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
import time
import uuid
from typing import Optional

import httpx
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Default SDXL workflow template for ComfyUI API format
SDXL_WORKFLOW_TEMPLATE = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 7.0,
            "denoise": 1.0,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 0,
            "steps": 25,
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors",
        },
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024,
        },
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": "",
        },
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": "",
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2],
        },
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "api_output",
            "images": ["8", 0],
        },
    },
}


class ImageService:
    """Handles text-to-image generation via dual ComfyUI instances."""

    def __init__(self):
        self._clients: dict[int, httpx.AsyncClient] = {}
        self._is_ready: dict[int, bool] = {}

    async def _get_client(self, device_id: int) -> httpx.AsyncClient:
        if device_id not in self._clients or self._clients[device_id].is_closed:
            url = settings.get_comfyui_url(device_id)
            self._clients[device_id] = httpx.AsyncClient(
                base_url=url,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._clients[device_id]

    async def health_check(self, device_id: int = 0) -> bool:
        """Check if a specific ComfyUI instance is responsive."""
        try:
            client = await self._get_client(device_id)
            resp = await client.get("/system_stats")
            ready = resp.status_code == 200
            self._is_ready[device_id] = ready
            return ready
        except Exception as e:
            logger.warning(f"ComfyUI GPU {device_id} health check failed: {e}")
            self._is_ready[device_id] = False
            return False

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        cfg_scale: float = 7.0,
        seed: int = -1,
        batch_size: int = 1,
        device_id: int = 0, # Added device_id
    ) -> dict:
        """Generate images using SDXL on a specific GPU's ComfyUI instance."""
        start_time = time.time()
        client = await self._get_client(device_id)

        # Resolve seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        # Build workflow from template
        workflow = json.loads(json.dumps(SDXL_WORKFLOW_TEMPLATE))
        workflow["3"]["inputs"]["seed"] = seed
        workflow["3"]["inputs"]["steps"] = steps
        workflow["3"]["inputs"]["cfg"] = cfg_scale
        workflow["4"]["inputs"]["ckpt_name"] = settings.sdxl_checkpoint
        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height
        workflow["5"]["inputs"]["batch_size"] = batch_size
        workflow["6"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = negative_prompt

        # Generate unique client ID
        client_id = str(uuid.uuid4())

        # Queue the prompt
        payload = {
            "prompt": workflow,
            "client_id": client_id,
        }

        logger.info(f"Queueing SDXL on GPU {device_id}: '{prompt[:40]}...'")

        resp = await client.post("/prompt", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI prompt queue failed (GPU {device_id}): {resp.status_code} — {resp.text}")

        prompt_id = resp.json()["prompt_id"]

        # Poll for completion
        images_b64 = await self._poll_and_retrieve(client, prompt_id)

        elapsed = time.time() - start_time
        logger.info(f"Generation on GPU {device_id} complete: {len(images_b64)} image(s) in {elapsed:.1f}s")

        return {
            "images": images_b64,
            "seed": seed,
            "prompt": prompt,
            "generation_time_seconds": round(elapsed, 2),
            "device_id": device_id,
        }

    async def _poll_and_retrieve(
        self,
        client: httpx.AsyncClient,
        prompt_id: str,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> list[str]:
        """Poll ComfyUI history for completion and retrieve images."""
        import asyncio

        start = time.time()
        while time.time() - start < timeout:
            resp = await client.get(f"/history/{prompt_id}")
            if resp.status_code == 200:
                history = resp.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    return await self._extract_images(client, outputs)

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"ComfyUI generation timed out after {timeout}s")

    async def _extract_images(
        self,
        client: httpx.AsyncClient,
        outputs: dict,
    ) -> list[str]:
        """Extract and download generated images from ComfyUI outputs."""
        images_b64 = []

        for node_id, node_output in outputs.items():
            if "images" not in node_output:
                continue

            for img_info in node_output["images"]:
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")

                params = {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": img_type,
                }
                resp = await client.get("/view", params=params)

                if resp.status_code == 200:
                    img_bytes = resp.content
                    b64_str = base64.b64encode(img_bytes).decode("utf-8")
                    images_b64.append(b64_str)
                else:
                    logger.error(f"Failed to download image {filename}: {resp.status_code}")

        return images_b64

    async def close(self):
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        self._clients = {}


# Global singleton
image_service = ImageService()
