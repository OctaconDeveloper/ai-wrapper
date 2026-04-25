from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Optional, Callable

import torch

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    IMAGE = "image"
    VIDEO_T2V = "video_t2v"
    VIDEO_I2V = "video_i2v"
    TEXT = "text"
    AUDIO = "audio"
    LSTM = "lstm"


class ModelState:
    """Tracks the state of a loaded model on a specific GPU."""

    def __init__(self, name: str, model_type: ModelType, device_id: int = 0):
        self.name = name
        self.model_type = model_type
        self.device_id = device_id
        self.instance: Any = None
        self.is_loaded: bool = False
        self.is_loading: bool = False
        self.last_used: float = 0.0
        self.vram_estimate_mb: float = 0.0
        self.error: Optional[str] = None
        self.unload_callback: Optional[Callable[[int], None]] = None

    def mark_loaded(self, instance: Any, vram_mb: float = 0.0, unload_callback: Optional[Callable[[int], None]] = None):
        self.instance = instance
        self.is_loaded = True
        self.is_loading = False
        self.last_used = time.time()
        self.vram_estimate_mb = vram_mb
        self.error = None
        self.unload_callback = unload_callback

    def mark_unloaded(self):
        self.instance = None
        self.is_loaded = False
        self.is_loading = False
        self.vram_estimate_mb = 0.0
        self.unload_callback = None

    def mark_error(self, error: str):
        self.is_loading = False
        self.error = error

    def touch(self):
        """Update last-used timestamp."""
        self.last_used = time.time()


def is_cuda_available() -> bool:
    """Robustly check for CUDA availability without throwing on CPU-only builds."""
    try:
        return torch.cuda.is_available()
    except (AssertionError, RuntimeError, Exception):
        return False

class ModelManager:
    """
    Manages lazy loading and VRAM for all models across multiple GPUs.
    Supports model duplication (same model on multiple GPUs).
    """

    def __init__(self, max_loaded_per_gpu: int = 3):
        self.max_loaded_per_gpu = max_loaded_per_gpu
        # Key: (model_type, device_id)
        self._models: dict[tuple[ModelType, int], ModelState] = {}
        self._lock = asyncio.Lock()
        
        # Improved device detection with safety wrappers
        cuda_ok = is_cuda_available()
        if cuda_ok:
            try:
                self._device_count = torch.cuda.device_count()
            except Exception:
                self._device_count = 1
                cuda_ok = False
        else:
            self._device_count = 1 # Fallback to 1 "device" for CPU
            
        logger.info(f"ModelManager initialized with {self._device_count} {'GPU(s)' if cuda_ok else 'CPU'}")

    @property
    def device_count(self) -> int:
        return self._device_count

    def get_device_string(self, device_id: int = 0) -> str:
        """Return 'cuda:x' or 'cpu' based on availability."""
        if is_cuda_available():
            return f"cuda:{device_id}"
        return "cpu"

    def register(self, model_type: ModelType, name: str, device_id: int = 0) -> ModelState:
        """Register a model for tracking on a specific GPU."""
        state = ModelState(name=name, model_type=model_type, device_id=device_id)
        self._models[(model_type, device_id)] = state
        logger.info(f"Registered model {name} on cuda:{device_id}")
        return state

    def get_state(self, model_type: ModelType, device_id: int = 0) -> Optional[ModelState]:
        return self._models.get((model_type, device_id))

    def get_all_states(self) -> dict[str, Any]:
        """Return status of all registered models across all GPUs."""
        result = {}
        for (mt, dev_id), state in self._models.items():
            status = "error" if state.error else (
                "loading" if state.is_loading else (
                    "loaded" if state.is_loaded else "unloaded"
                )
            )
            key = f"{mt.value}_gpu{dev_id}"
            result[key] = {
                "name": state.name,
                "model_type": mt.value,
                "device_id": dev_id,
                "status": status,
                "vram_mb": state.vram_estimate_mb,
                "last_used": state.last_used,
                "error": state.error,
            }
        return result

    def get_loaded_count(self, device_id: int) -> int:
        return sum(1 for (mt, dev), s in self._models.items() if dev == device_id and s.is_loaded)

    async def ensure_capacity(self, device_id: int, exclude_type: Optional[ModelType] = None):
        """Evict LRU model on a specific GPU if max capacity is reached."""
        async with self._lock:
            while self.get_loaded_count(device_id) >= self.max_loaded_per_gpu:
                # Find LRU candidate on this GPU
                candidates = [
                    (mt, s) for (mt, dev), s in self._models.items()
                    if dev == device_id and s.is_loaded and mt != exclude_type
                ]
                
                if not candidates:
                    logger.warning(f"GPU {device_id} is full but no eviction candidates found (excluding {exclude_type})")
                    break
                
                candidates.sort(key=lambda x: x[1].last_used)
                mt_to_evict, state_to_evict = candidates[0]
                
                logger.info(f"Evicting {mt_to_evict.value} from GPU {device_id} to free capacity")
                
                if state_to_evict.unload_callback:
                    try:
                        # Call the service-provided unload logic
                        state_to_evict.unload_callback(device_id)
                    except Exception as e:
                        logger.error(f"Error during unload callback for {mt_to_evict.value} on GPU {device_id}: {e}")
                
                # Cleanup
                state_to_evict.mark_unloaded()
                self.clear_gpu_cache(device_id)

    def find_best_gpu(self, model_type: ModelType) -> int:
        """
        Pick the best GPU for a task:
        1. GPU where the model is already loaded.
        2. GPU with the most free VRAM / least loaded models.
        """
        # Priority 1: Already loaded
        for dev_id in range(self._device_count):
            state = self.get_state(model_type, dev_id)
            if state and state.is_loaded:
                return dev_id
        
        # Priority 2: Least number of models currently loaded
        counts = [(dev_id, self.get_loaded_count(dev_id)) for dev_id in range(self._device_count)]
        counts.sort(key=lambda x: x[1])
        return counts[0][0]

    @staticmethod
    def get_gpu_memory_info(device_id: int = 0) -> dict:
        """Get current GPU memory usage for a specific device."""
        try:
            if not is_cuda_available() or device_id >= torch.cuda.device_count():
                return {"used_mb": 0, "total_mb": 0, "free_mb": 0}

            props = torch.cuda.get_device_properties(device_id)
            # We use memory_allocated for "used" and memory_reserved for what torch holds
            used = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            cached = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
            total = props.total_mem / (1024 ** 2)

            return {
                "device": props.name,
                "used_mb": round(used, 1),
                "cached_mb": round(cached, 1),
                "total_mb": round(total, 1),
                "free_mb": round(total - cached, 1),
            }
        except Exception:
            return {"used_mb": 0, "total_mb": 0, "free_mb": 0}

    @staticmethod
    def clear_gpu_cache(device_id: int):
        """Force clear CUDA memory cache for a specific device."""
        try:
            if is_cuda_available() and device_id < torch.cuda.device_count():
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info(f"GPU {device_id} cache cleared")
        except Exception:
            pass


# Global singleton
model_manager = ModelManager()
