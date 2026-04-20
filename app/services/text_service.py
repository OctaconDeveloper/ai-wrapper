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

from app.config import settings
from app.services.model_manager import ModelType, model_manager, ModelManager

logger = logging.getLogger(__name__)


class TextService:
    """Handles text generation via Dolphin-Mixtral 8x7B (GGUF)."""

    def __init__(self):
        self._llm = None
        self._state = model_manager.register(ModelType.TEXT, "Dolphin-Mixtral 8x7B (Q4_K_M)")

    def _load_model(self):
        """Lazy-load the GGUF model."""
        if self._llm is not None:
            self._state.touch()
            return

        logger.info(f"Loading Mixtral GGUF: {settings.mixtral_model_path}")
        self._state.is_loading = True

        try:
            from llama_cpp import Llama

            if model_manager.should_evict():
                candidate = model_manager.get_eviction_candidate(exclude=ModelType.TEXT)
                if candidate:
                    logger.info(f"Evicting {candidate.value} before loading LLM")
                    ModelManager.clear_gpu_cache()

            self._llm = Llama(
                model_path=settings.mixtral_model_path,
                n_ctx=settings.mixtral_context_length,
                n_gpu_layers=settings.mixtral_gpu_layers,
                n_threads=8,
                verbose=False,
                flash_attn=True,
            )

            self._state.mark_loaded(self._llm, vram_mb=26000)
            logger.info("Dolphin-Mixtral loaded successfully")

        except Exception as e:
            self._state.mark_error(str(e))
            logger.error(f"Failed to load Mixtral: {e}")
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
    ) -> dict | AsyncGenerator:
        """
        Generate text using Dolphin-Mixtral.

        Supports both raw completion (prompt) and chat completion (messages).
        """
        self._load_model()

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        if stream:
            return self._stream_generate(
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

        # Run blocking llama-cpp inference in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            self._sync_generate,
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
        logger.info(f"Generation complete in {elapsed:.1f}s")

        result["generation_time_seconds"] = round(elapsed, 2)
        return result

    def _sync_generate(
        self,
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
            logger.info(f"Chat completion: {len(messages)} messages, max_tokens={max_tokens}")
            result = self._llm.create_chat_completion(
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
            logger.info(f"Completion: '{prompt[:80]}...', max_tokens={max_tokens}")
            result = self._llm(
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
            stream = self._llm.create_chat_completion(
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
            stream = self._llm(
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

    def unload(self):
        """Unload the model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._state.mark_unloaded()
            ModelManager.clear_gpu_cache()
            logger.info("Mixtral model unloaded")


# Global singleton
text_service = TextService()
