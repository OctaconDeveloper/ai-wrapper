"""
Text Service — Dolphin-Mixtral 8x7B text generation via llama-cpp-python.

Loads the GGUF quantized model directly, supports both completion and chat modes,
with optional SSE streaming. Fully uncensored — no safety filters.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import AsyncGenerator, Optional

import torch
from app.config import settings
from app.services.model_manager import ModelType, model_manager, ModelManager, is_cuda_available

logger = logging.getLogger(__name__)


class TextService:
    """Handles text generation via Dolphin-Mixtral 8x7B (GGUF)."""

    def __init__(self):
        # We track models per device
        self._states: dict[int, Any] = {}

    def _load_model(self, device_id: int):
        """Lazy-load the GGUF model on a specific GPU."""
        device_str = model_manager.get_device_string(device_id)
        state = model_manager.get_state(ModelType.TEXT, device_id)
        if not state:
            state = model_manager.register(ModelType.TEXT, f"Dolphin-Mistral-Nemo 12B ({device_str})", device_id)

        if state.instance is not None:
            state.touch()
            return state.instance

        import os
        if not os.path.exists(settings.mixtral_model_path):
            error_msg = f"Model file NOT FOUND at: {settings.mixtral_model_path}. Please check your volume mounts."
            logger.error(error_msg)
            state.mark_error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading Dolphin-Mistral-Nemo GGUF on {device_str}: {settings.mixtral_model_path}")
        state.is_loading = True

        try:
            from llama_cpp import Llama

            # On Mac/CPU, disabling mmap can sometimes be more stable for very large files in Docker
            use_mmap = True
            if settings.mixtral_gpu_layers == 0 or not is_cuda_available():
                logger.info("CPU mode detected: using 8 threads and disabling mmap for stability.")
                use_mmap = False 

            llm = Llama(
                model_path=settings.mixtral_model_path,
                n_ctx=settings.mixtral_context_length,
                n_gpu_layers=settings.mixtral_gpu_layers if is_cuda_available() else 0,
                n_threads=8,
                verbose=True, # Enable verbose for better llama.cpp logs
                use_mmap=use_mmap,
                flash_attn=True if (settings.mixtral_gpu_layers > 0 and is_cuda_available()) else False,
                main_gpu=device_id if is_cuda_available() else 0,
            )

            state.mark_loaded(llm, vram_mb=26000, unload_callback=self.unload)
            logger.info(f"Dolphin-Mixtral loaded on {device_str} successfully")
            return llm

        except Exception as e:
            state.mark_error(str(e))
            logger.error(f"Failed to load Mixtral on {device_str}: {e}")
            raise

    async def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: int = -1,
        stream: bool = False,
        device_id: int = 0,  # Added device_id
    ) -> dict | AsyncGenerator:
        """
        Generate text using Dolphin-Mixtral.
        """
        # Ensure model is loaded on the assigned device
        await model_manager.ensure_capacity(device_id, exclude_type=ModelType.TEXT)
        llm = self._load_model(device_id)

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        if stream:
            return self._stream_generate(
                llm=llm,
                prompt=prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                seed=seed,
            )

        start_time = time.time()

        # Run blocking llama-cpp inference in a thread
        result = await asyncio.to_thread(
            self._sync_generate,
            llm=llm,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        elapsed = time.time() - start_time
        logger.info(f"Generation on GPU {device_id} complete in {elapsed:.1f}s")

        result["generation_time_seconds"] = round(elapsed, 2)
        result["device_id"] = device_id
        return result

    def _sync_generate(
        self,
        llm: Any,
        prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: int = -1,
    ) -> dict:
        """Synchronous generation — called from a thread."""
        if messages:
            result = llm.create_chat_completion(
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                seed=seed,
            )
            text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
        elif prompt:
            result = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                seed=seed,
            )
            text = result["choices"][0]["text"]
            usage = result.get("usage", {})
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        return {
            "text": text,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    async def _stream_generate(
        self,
        llm: Any,
        prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: int = -1,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens via SSE."""
        import json

        if messages:
            stream = llm.create_chat_completion(
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                seed=seed,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield f"data: {json.dumps({'text': content})}\n\n"
        elif prompt:
            stream = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                seed=seed,
                stream=True,
            )
            for chunk in stream:
                text = chunk["choices"][0]["text"]
                if text:
                    yield f"data: {json.dumps({'text': text})}\n\n"

        yield "data: [DONE]\n\n"

    def unload(self, device_id: int):
        """Unload the model from memory on a specific GPU."""
        state = model_manager.get_state(ModelType.TEXT, device_id)
        if state and state.instance is not None:
            # For llama-cpp, we just delete the instance
            del state.instance
            state.mark_unloaded()
            logger.info(f"Mixtral model unloaded from GPU {device_id}")


# Global singleton
text_service = TextService()
