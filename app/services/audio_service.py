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
        self._tts = None
        self._state = model_manager.register(ModelType.AUDIO, "XTTS v2")
        self._default_speaker_path: Optional[str] = None

    def _load_model(self):
        """Lazy-load XTTS v2 model."""
        if self._tts is not None:
            self._state.touch()
            return

        logger.info(f"Loading XTTS v2: {settings.xtts_model}")
        self._state.is_loading = True

        try:
            from TTS.api import TTS

            if model_manager.should_evict():
                candidate = model_manager.get_eviction_candidate(exclude=ModelType.AUDIO)
                if candidate:
                    logger.info(f"Evicting {candidate.value} before loading XTTS")
                    ModelManager.clear_gpu_cache()

            self._tts = TTS(model_name=settings.xtts_model, gpu=True)

            # Create a default speaker reference (silent WAV) for cases without voice cloning
            self._ensure_default_speaker()

            self._state.mark_loaded(self._tts, vram_mb=2000)
            logger.info("XTTS v2 loaded successfully")

        except Exception as e:
            self._state.mark_error(str(e))
            logger.error(f"Failed to load XTTS v2: {e}")
            raise

    def _ensure_default_speaker(self):
        """Create a minimal default speaker WAV if none exists."""
        default_path = Path(settings.output_path) / "default_speaker.wav"
        if not default_path.exists():
            # Generate a brief silent WAV as a fallback
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
    ) -> dict:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            language: Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, hi)
            speaker_wav_base64: Optional base64-encoded WAV for voice cloning
            speed: Speech speed multiplier (0.5 to 2.0)

        Returns:
            Dict with audio_base64, timing info, etc.
        """
        start_time = time.time()

        self._load_model()

        # Handle speaker reference
        speaker_path = self._default_speaker_path
        tmp_speaker = None

        if speaker_wav_base64:
            # Save uploaded speaker WAV to temp file
            speaker_bytes = base64.b64decode(speaker_wav_base64)
            tmp_speaker = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_speaker.write(speaker_bytes)
            tmp_speaker.close()
            speaker_path = tmp_speaker.name

        try:
            logger.info(f"Generating speech: '{text[:80]}...' (lang={language}, speed={speed})")

            # Generate to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                output_path = tmp_out.name

            self._tts.tts_to_file(
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
            logger.info(f"Speech generated in {elapsed:.1f}s ({duration:.1f}s audio)")

            return {
                "audio_base64": audio_b64,
                "text": text,
                "language": language,
                "duration_seconds": round(duration, 2),
                "generation_time_seconds": round(elapsed, 2),
            }

        finally:
            # Cleanup temp files
            Path(output_path).unlink(missing_ok=True)
            if tmp_speaker:
                Path(tmp_speaker.name).unlink(missing_ok=True)

    def unload(self):
        """Unload the model from memory."""
        if self._tts is not None:
            del self._tts
            self._tts = None
            self._state.mark_unloaded()
            ModelManager.clear_gpu_cache()
            logger.info("XTTS v2 model unloaded")


# Global singleton
audio_service = AudioService()
