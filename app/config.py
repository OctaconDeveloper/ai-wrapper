"""
Configuration module — centralizes all settings and model paths.
Uses pydantic-settings for environment variable binding.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Server ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    comfyui_port: int = Field(default=8188, alias="COMFYUI_PORT")
    comfyui_host: str = Field(default="127.0.0.1", alias="COMFYUI_HOST")

    # --- Paths ---
    models_dir: str = Field(default="/workspace/models", alias="MODELS_DIR")
    comfyui_dir: str = Field(default="/workspace/ComfyUI", alias="COMFYUI_DIR")
    output_dir: str = Field(default="/workspace/outputs", alias="OUTPUT_DIR")

    # --- SDXL / ComfyUI ---
    sdxl_checkpoint: str = Field(
        default="sd_xl_base_1.0.safetensors",
        alias="SDXL_CHECKPOINT",
    )

    # --- Mixtral / LLM ---
    mixtral_model_path: str = Field(
        default="/workspace/models/llm/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
        alias="MIXTRAL_MODEL_PATH",
    )
    mixtral_context_length: int = Field(default=32768, alias="MIXTRAL_CONTEXT_LENGTH")
    mixtral_gpu_layers: int = Field(default=-1, alias="MIXTRAL_GPU_LAYERS")

    # --- Wan 2.1 Video ---
    wan_t2v_model: str = Field(
        default="/workspace/models/video/wan2.1-t2v-1.3b",
        alias="WAN_T2V_MODEL",
    )
    wan_i2v_model: str = Field(
        default="/workspace/models/video/wan2.1-i2v-14b",
        alias="WAN_I2V_MODEL",
    )

    # --- XTTS v2 ---
    xtts_model: str = Field(
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        alias="XTTS_MODEL",
    )

    # --- GPU Memory Management ---
    max_loaded_models: int = Field(default=3, alias="MAX_LOADED_MODELS")
    enable_model_offload: bool = Field(default=True, alias="ENABLE_MODEL_OFFLOAD")

    # --- Auto-Shutdown (cost saving for per hour GPU) ---
    idle_shutdown_enabled: bool = Field(
        default=True,
        alias="IDLE_SHUTDOWN_ENABLED",
        description="Auto-stop instance after idle timeout",
    )
    idle_shutdown_minutes: int = Field(
        default=30,
        alias="IDLE_SHUTDOWN_MINUTES",
        description="Minutes of inactivity before auto-shutdown",
    )
    vastai_api_key: str = Field(
        default="",
        alias="VASTAI_API_KEY",
        description="Vast.ai API key for programmatic instance stop",
    )
    vastai_instance_id: str = Field(
        default="",
        alias="VASTAI_INSTANCE_ID",
        description="Current vast.ai instance ID (auto-detected if empty)",
    )

    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"

    @property
    def comfyui_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def sdxl_checkpoint_path(self) -> Path:
        return Path(self.comfyui_dir) / "models" / "checkpoints" / self.sdxl_checkpoint

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Singleton settings instance
settings = Settings()
