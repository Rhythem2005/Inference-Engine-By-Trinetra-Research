"""Thread-safe CUDA and CPU inference engine for KAAL-304M on Windows."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
from tokenizers import Tokenizer

from backend.engine.config import (
    DEFAULT_DECODING_CONFIG,
    DEFAULT_MODEL_CONFIG,
    EOS_TOKEN,
    EXPECTED_UNIQUE_PARAMS,
    MODEL_NAME,
    DecodingConfig,
    ModelConfig,
    format_prompt,
)
from backend.engine.inference import sample_next_token
from backend.engine.model import KAALModel, count_unique_parameters

logger = logging.getLogger(__name__)


class RetryOnCPU(RuntimeError):
    """CUDA memory was exhausted after streaming began. Resend the request on CPU."""


def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """Use NVIDIA CUDA when available, keeping the reference FP32 precision."""
    preference = os.environ.get("KAAL_DEVICE", "auto").strip().lower()
    if preference not in {"auto", "cpu"}:
        raise ValueError("KAAL_DEVICE must be auto or cpu")
    if preference == "auto" and torch.version.cuda is not None:
        try:
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device()), torch.float32
        except (RuntimeError, OSError) as exc:
            logger.warning("CUDA initialization failed. Using CPU: %s", exc)
    return torch.device("cpu"), torch.float32


class KAALEngine:
    """Thread-safe local inference engine for KAAL-304M on Windows."""

    def __init__(
        self,
        weights_dir: str | Path = "weights",
        model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
        *,
        decoding_config: DecodingConfig = DEFAULT_DECODING_CONFIG,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.checkpoint_path = self.weights_dir / "best_model.pt"
        self.tokenizer_path = self.weights_dir / "tokenizer.json"
        self.model_config = model_config
        self.decoding_config = decoding_config
        if decoding_config.max_new_tokens >= model_config.context_length:
            raise ValueError("max_new_tokens must be smaller than the model context length")

        self.device, self.dtype = select_device_and_dtype()
        self.model: Optional[KAALModel] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.eos_id: int = 0
        self.unique_params: int = EXPECTED_UNIQUE_PARAMS
        self.weights_loaded: bool = False
        self.status: str = "loading"
        self._lock = threading.Lock()

    def missing_files(self) -> List[str]:
        """Return list of missing required weight files."""
        missing: List[str] = []
        if not self.checkpoint_path.is_file():
            missing.append("weights/best_model.pt")
        if not self.tokenizer_path.is_file():
            missing.append("weights/tokenizer.json")
        return missing

    def health_status(self) -> Dict[str, Any]:
        """Return structured engine health metadata for /api/health."""
        missing = self.missing_files()
        if missing:
            status = "weights_missing"
        elif self.weights_loaded:
            status = "ready"
        else:
            status = self.status
        return {
            "status": status,
            "model_name": MODEL_NAME,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "params": self.unique_params,
            "weights_loaded": self.weights_loaded,
            "missing_files": missing,
            "max_new_tokens": self.decoding_config.max_new_tokens,
        }

    def load(self) -> bool:
        """Load tokenizer and checkpoint strictly, move to device, and run 1-token warmup."""
        with self._lock:
            missing = self.missing_files()
            if missing:
                self.weights_loaded = False
                self.status = "weights_missing"
                return False

            self.status = "loading"
            tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            if set(tokenizer.get_vocab().values()) != set(range(self.model_config.vocab_size)):
                raise ValueError("Tokenizer token IDs do not match the model vocabulary")
            eos_id = tokenizer.token_to_id(EOS_TOKEN)
            if eos_id is None:
                raise ValueError(f"Tokenizer is missing required EOS token {EOS_TOKEN!r}")

            model = KAALModel(self.model_config)
            # Accept tensor state dicts without enabling arbitrary pickle deserialization.
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            model.load_state_dict(state_dict, strict=True)

            device = self.device
            try:
                unique_params = self._prepare_model(model, device, eos_id)
            except (RuntimeError, OSError) as exc:
                if device.type != "cuda":
                    raise
                logger.warning("CUDA model setup failed. Using CPU: %s", exc)
                device = torch.device("cpu")
            if device != self.device:
                # A failed CUDA transfer can leave the model partly on the GPU.
                del model
                with suppress(RuntimeError, OSError):
                    torch.cuda.empty_cache()
                model = KAALModel(self.model_config)
                model.load_state_dict(state_dict, strict=True)
                unique_params = self._prepare_model(model, device, eos_id)

            self.device = device
            self.tokenizer = tokenizer
            self.eos_id = int(eos_id)
            self.model = model
            self.unique_params = unique_params
            self.weights_loaded = True
            self.status = "ready"
            return True

    def _move_model(self, model: KAALModel, device: torch.device) -> None:
        model.to(device=device, dtype=self.dtype)
        model.out_head.weight = model.tok_emb.weight
        if model.out_head.weight.data_ptr() != model.tok_emb.weight.data_ptr():
            raise RuntimeError("Weight tying broken after moving KAALModel to device")

    def _prepare_model(self, model: KAALModel, device: torch.device, eos_id: int) -> int:
        self._move_model(model, device)
        model.eval()
        unique_params = count_unique_parameters(model)
        if unique_params != EXPECTED_UNIQUE_PARAMS:
            raise RuntimeError(
                f"Unexpected unique parameter count: {unique_params} != {EXPECTED_UNIQUE_PARAMS}"
            )

        # Run the reference one-token warmup and surface asynchronous CUDA errors.
        with torch.inference_mode():
            dummy = torch.tensor([[eos_id]], dtype=torch.long, device=device)
            model(dummy, past_kv=None, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return unique_params

    def stream(
        self,
        instruction: str,
        context: str = "",
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream UTF-8 BPE text deltas and final generation telemetry."""
        with self._lock:
            if not self.weights_loaded or self.model is None or self.tokenizer is None:
                raise RuntimeError("KAAL-304M weights are not loaded.")

            emitted = False
            try:
                for event in self._stream(instruction, context, cancel_event):
                    emitted = True
                    yield event
                return
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                logger.warning("CUDA ran out of memory during generation. Switching to CPU.")

            # Leave the exception scope before moving the model so its traceback
            # no longer holds the failed request's CUDA tensors.
            cpu = torch.device("cpu")
            self._move_model(self.model, cpu)
            self.device = cpu
            torch.cuda.empty_cache()
            if emitted:
                raise RetryOnCPU("GPU memory ran out. KAAL switched to CPU. Please resend your message.")
            yield from self._stream(instruction, context, cancel_event)

    def _stream(
        self,
        instruction: str,
        context: str,
        cancel_event: Optional[threading.Event],
    ) -> Generator[Dict[str, Any], None, None]:
        decoding = self.decoding_config
        prompt = format_prompt(instruction, context)
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids: List[int] = list(encoded.ids)
        if not prompt_ids:
            prompt_ids = [self.eos_id]

        max_prompt_tokens = self.model_config.context_length - decoding.max_new_tokens
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        generated_ids: List[int] = []
        emitted_text = ""
        stopped_reason = "max_tokens"
        start_time = time.perf_counter()

        with torch.inference_mode():
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            logits, past_kv = self.model(input_ids, past_kv=None, use_cache=True)
            next_logits = logits[0, -1, :].float()

            for _ in range(decoding.max_new_tokens):
                if cancel_event is not None and cancel_event.is_set():
                    stopped_reason = "cancelled"
                    break

                next_token = sample_next_token(next_logits, generated_ids, decoding)
                if next_token == self.eos_id:
                    stopped_reason = "eos"
                    break

                generated_ids.append(next_token)
                decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Hold back incomplete multi-byte UTF-8 sequences until completed.
                if (
                    "\ufffd" not in decoded[len(emitted_text) :]
                    and len(decoded) > len(emitted_text)
                    and decoded.startswith(emitted_text)
                ):
                    delta = decoded[len(emitted_text) :]
                    emitted_text = decoded
                    yield {"type": "token", "text": delta}

                if len(generated_ids) >= decoding.max_new_tokens:
                    stopped_reason = "max_tokens"
                    break

                if cancel_event is not None and cancel_event.is_set():
                    stopped_reason = "cancelled"
                    break

                step_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                logits, past_kv = self.model(step_input, past_kv=past_kv, use_cache=True)
                next_logits = logits[0, -1, :].float()

        # Flush any trailing decoded text (e.g. if stopped while buffering UTF-8).
        if generated_ids:
            final_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if len(final_decoded) > len(emitted_text) and final_decoded.startswith(emitted_text):
                tail = final_decoded[len(emitted_text) :]
                if tail:
                    yield {"type": "token", "text": tail}

        elapsed = max(time.perf_counter() - start_time, 1e-6)
        token_count = len(generated_ids)
        tps = round(token_count / elapsed, 2) if token_count > 0 else 0.0

        yield {
            "type": "done",
            "tokens": token_count,
            "tokens_per_second": tps,
            "stopped_reason": stopped_reason,
        }

    def generate(self, instruction: str, context: str = "") -> Dict[str, Any]:
        """Generate a complete response synchronously using the configured profile."""
        chunks: List[str] = []
        done_meta: Dict[str, Any] = {
            "tokens": 0,
            "tokens_per_second": 0.0,
            "stopped_reason": "eos",
        }
        for event in self.stream(instruction=instruction, context=context):
            if event["type"] == "token":
                chunks.append(event["text"])
            elif event["type"] == "done":
                done_meta = event
        return {
            "text": "".join(chunks),
            "tokens": done_meta["tokens"],
            "tokens_per_second": done_meta["tokens_per_second"],
            "stopped_reason": done_meta["stopped_reason"],
        }
