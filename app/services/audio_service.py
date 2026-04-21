"""
Audio Service — XTTS v2 text-to-speech with voice cloning.

Uses the Coqui TTS library to load XTTS v2 for multi-language synthesis.
Supports optional voice cloning from a reference WAV file.
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings
from app.services.model_manager import ModelType, model_manager, ModelManager

logger = logging.getLogger(__name__)

# Default reference speaker audio — bundled with XTTS v2
DEFAULT_SPEAKER_WAV = None


class AudioService:
    """Handles text-to-speech via XTTS v2."""

    def __init__(self):
        # State per device
        self._states: dict[int, Any] = {}
        self._default_speaker_path: Optional[str] = None

    def _load_model(self, device_id: int):
        """Lazy-load XTTS v2 model on a specific GPU."""
        state = model_manager.get_state(ModelType.AUDIO, device_id)
        if not state:
            state = model_manager.register(ModelType.AUDIO, f"XTTS v2 (GPU {device_id})", device_id)

        if state.instance is not None:
            state.touch()
            return state.instance

        logger.info(f"Loading XTTS v2 on cuda:{device_id}: {settings.xtts_model}")
        state.is_loading = True

        try:
            from TTS.api import TTS

            # Use GPU for synthesis
            tts = TTS(model_name=settings.xtts_model, gpu=True)
            # Pin to specific device if supported by the library
            if hasattr(tts, "to"):
                tts.to(f"cuda:{device_id}")

            # Create a default speaker reference (silent WAV)
            self._ensure_default_speaker()

            state.mark_loaded(tts, vram_mb=2000, unload_callback=self.unload)
            logger.info(f"XTTS v2 loaded on GPU {device_id} successfully")
            return tts

        except Exception as e:
            state.mark_error(str(e))
            logger.error(f"Failed to load XTTS v2 on GPU {device_id}: {e}")
            raise

    def _ensure_default_speaker(self):
        """Create a minimal default speaker WAV if none exists."""
        default_path = Path(settings.output_path) / "default_speaker.wav"
        if not default_path.exists():
            sample_rate = 22050
            duration = 1.0
            samples = np.zeros(int(sample_rate * duration), dtype=np.int16)

            with wave.open(str(default_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())

        self._default_speaker_path = str(default_path)

    async def generate(
        self,
        text: str,
        language: str = "en",
        speaker_wav_base64: Optional[str] = None,
        speed: float = 1.0,
        device_id: int = 0, # Added device_id
    ) -> dict:
        """Generate speech from text."""
        start_time = time.time()

        # Ensure capacity and load model
        await model_manager.ensure_capacity(device_id, exclude_type=ModelType.AUDIO)
        tts = self._load_model(device_id)

        # Handle speaker reference
        speaker_path = self._default_speaker_path
        tmp_speaker_file = None

        if speaker_wav_base64:
            speaker_bytes = base64.b64decode(speaker_wav_base64)
            tmp_speaker_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_speaker_file.write(speaker_bytes)
            tmp_speaker_file.close()
            speaker_path = tmp_speaker_file.name

        try:
            logger.info(f"Generating speech on GPU {device_id}: '{text[:80]}...'")

            # Generate to temp file (non-blocking thread)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                output_path = tmp_out.name

            await asyncio.to_thread(
                tts.tts_to_file,
                text=text,
                file_path=output_path,
                speaker_wav=speaker_path,
                language=language,
                speed=speed,
            )

            # Read the output WAV
            with open(output_path, "rb") as f:
                audio_bytes = f.read()

            # Calculate duration
            with wave.open(output_path, "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            elapsed = time.time() - start_time
            logger.info(f"Speech generated on GPU {device_id} in {elapsed:.1f}s")

            return {
                "audio_base64": audio_b64,
                "text": text,
                "language": language,
                "duration_seconds": round(duration, 2),
                "generation_time_seconds": round(elapsed, 2),
                "device_id": device_id,
            }

        finally:
            # Cleanup temp files
            if'output_path' in locals():
                Path(output_path).unlink(missing_ok=True)
            if tmp_speaker_file:
                Path(tmp_speaker_file.name).unlink(missing_ok=True)

    def unload(self, device_id: int):
        """Unload the model from memory."""
        state = model_manager.get_state(ModelType.AUDIO, device_id)
        if state and state.instance is not None:
            del state.instance
            state.mark_unloaded()
            logger.info(f"XTTS v2 model unloaded from GPU {device_id}")


# Global singleton
audio_service = AudioService()
