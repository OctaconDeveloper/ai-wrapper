"""
Model Manager — handles lazy loading, GPU memory tracking, and model offloading.

Models are loaded on first request and can be offloaded when VRAM is constrained.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    IMAGE = "image"
    VIDEO_T2V = "video_t2v"
    VIDEO_I2V = "video_i2v"
    TEXT = "text"
    AUDIO = "audio"


class ModelState:
    """Tracks the state of a loaded model."""

    def __init__(self, name: str, model_type: ModelType):
        self.name = name
        self.model_type = model_type
        self.instance: Any = None
        self.is_loaded: bool = False
        self.is_loading: bool = False
        self.last_used: float = 0.0
        self.vram_estimate_mb: float = 0.0
        self.error: Optional[str] = None

    def mark_loaded(self, instance: Any, vram_mb: float = 0.0):
        self.instance = instance
        self.is_loaded = True
        self.is_loading = False
        self.last_used = time.time()
        self.vram_estimate_mb = vram_mb
        self.error = None

    def mark_unloaded(self):
        self.instance = None
        self.is_loaded = False
        self.is_loading = False
        self.vram_estimate_mb = 0.0

    def mark_error(self, error: str):
        self.is_loading = False
        self.error = error

    def touch(self):
        """Update last-used timestamp."""
        self.last_used = time.time()


class ModelManager:
    """
    Manages lazy loading and VRAM for all models.
    Uses LRU eviction when max_loaded_models is exceeded.
    """

    def __init__(self, max_loaded_models: int = 3):
        self.max_loaded_models = max_loaded_models
        self._models: dict[ModelType, ModelState] = {}
        self._lock = None  # Initialized lazily with asyncio lock

    def register(self, model_type: ModelType, name: str) -> ModelState:
        """Register a model type for tracking."""
        state = ModelState(name=name, model_type=model_type)
        self._models[model_type] = state
        logger.info(f"Registered model: {name} ({model_type.value})")
        return state

    def get_state(self, model_type: ModelType) -> Optional[ModelState]:
        return self._models.get(model_type)

    def get_all_states(self) -> dict[str, dict]:
        """Return status of all registered models."""
        result = {}
        for mt, state in self._models.items():
            status = "error" if state.error else (
                "loading" if state.is_loading else (
                    "loaded" if state.is_loaded else "unloaded"
                )
            )
            result[mt.value] = {
                "name": state.name,
                "status": status,
                "vram_mb": state.vram_estimate_mb,
                "last_used": state.last_used,
                "error": state.error,
            }
        return result

    def get_loaded_count(self) -> int:
        return sum(1 for s in self._models.values() if s.is_loaded)

    def should_evict(self) -> bool:
        """Check if we need to evict a model before loading a new one."""
        return self.get_loaded_count() >= self.max_loaded_models

    def get_eviction_candidate(self, exclude: ModelType) -> Optional[ModelType]:
        """Find the least recently used loaded model (LRU eviction)."""
        candidates = [
            (mt, s) for mt, s in self._models.items()
            if s.is_loaded and mt != exclude
        ]
        if not candidates:
            return None
        # Sort by last_used ascending (oldest first)
        candidates.sort(key=lambda x: x[1].last_used)
        return candidates[0][0]

    @staticmethod
    def get_gpu_memory_info() -> dict:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"used_mb": 0, "total_mb": 0, "free_mb": 0}

        used = torch.cuda.memory_allocated() / (1024 ** 2)
        cached = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)

        return {
            "used_mb": round(used, 1),
            "cached_mb": round(cached, 1),
            "total_mb": round(total, 1),
            "free_mb": round(total - cached, 1),
        }

    @staticmethod
    def clear_gpu_cache():
        """Force clear CUDA memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")


# Global singleton
model_manager = ModelManager()
