import torch
import torch.nn as nn
import time
import logging
import asyncio
import os
from pathlib import Path
from typing import Optional

from app.config import settings
from app.services.model_manager import model_manager, ModelType

logger = logging.getLogger(__name__)


class SimpleLSTM(nn.Module):
    """A lightweight character-level LSTM model."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


class LSTMService:
    """Handles lightweight text generation via PyTorch LSTM."""

    def __init__(self):
        self._vocab = {chr(i): i for i in range(128)}  # Simple ASCII vocab
        self._inverse_vocab = {i: chr(i) for i in range(128)}

    def _load_model(self, device_id: int):
        """Lazy-load the LSTM model on a specific GPU."""
        device_str = model_manager.get_device_string(device_id)
        state = model_manager.get_state(ModelType.LSTM, device_id)
        if not state:
            state = model_manager.register(ModelType.LSTM, f"Lightweight LSTM ({device_str})", device_id)

        if state.instance is not None:
            state.touch()
            return state.instance

        logger.info(f"Loading LSTM on {device_str}: {settings.lstm_model_path}")
        state.is_loading = True

        try:
            model = SimpleLSTM(
                vocab_size=settings.lstm_vocab_size,
                embedding_dim=settings.lstm_embedding_dim,
                hidden_dim=settings.lstm_hidden_dim,
                num_layers=settings.lstm_num_layers,
            )

            # Check for weights, otherwise initialize with dummy data for testing
            path = Path(settings.lstm_model_path)
            if path.exists():
                model.load_state_dict(torch.load(path, map_location=device_str))
            else:
                logger.warning(f"LSTM weights not found at {path}. Initializing with random weights.")
                # Ensure directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                # We don't save here to avoid side effects, just use in-memory

            model.to(device_str)
            model.eval()

            state.mark_loaded(model, vram_mb=500, unload_callback=self.unload)
            logger.info(f"LSTM loaded on {device_str} successfully")
            return model

        except Exception as e:
            state.mark_error(str(e))
            logger.error(f"Failed to load LSTM on {device_str}: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        device_id: int = 0,
    ) -> dict:
        """Generate text character-by-character using the LSTM."""
        start_time = time.time()

        # Ensure GPU space and load model
        await model_manager.ensure_capacity(device_id, exclude_type=ModelType.LSTM)
        model = self._load_model(device_id)

        # Run inference in a thread to keep API responsive
        generated_text = await asyncio.to_thread(
            self._sync_generate,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            device_id=device_id
        )

        elapsed = time.time() - start_time
        logger.info(f"LSTM generation on {model_manager.get_device_string(device_id)} complete in {elapsed:.3f}s")

        return {
            "text": generated_text,
            "prompt": prompt,
            "prompt_tokens": len(prompt),
            "completion_tokens": len(generated_text) - len(prompt),
            "total_tokens": len(generated_text),
            "generation_time_seconds": round(elapsed, 3),
            "device_id": device_id,
        }

    def _sync_generate(self, model, prompt, max_tokens, temperature, device_id):
        """Synchronous character-level generation."""
        device = model_manager.get_device_string(device_id)
        input_ids = torch.tensor([[self._vocab.get(c, 0) for c in prompt]], device=device)
        
        generated = prompt
        hidden = None

        with torch.no_grad():
            # Warm up with prompt
            output, hidden = model(input_ids, hidden)
            
            # Predict next characters
            for _ in range(max_tokens):
                logits = output[:, -1, :] / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                
                char = self._inverse_vocab.get(next_id, "")
                generated += char
                
                # Prepare next input
                input_ids = torch.tensor([[next_id]], device=device)
                output, hidden = model(input_ids, hidden)

                if char == "\n" or len(generated) > 2048: # Basic safety
                    break

        return generated

    def unload(self, device_id: int):
        """Unload the LSTM from memory."""
        state = model_manager.get_state(ModelType.LSTM, device_id)
        if state and state.instance is not None:
            del state.instance
            state.mark_unloaded()
            logger.info(f"LSTM model unloaded from GPU {device_id}")


# Global singleton
lstm_service = LSTMService()
