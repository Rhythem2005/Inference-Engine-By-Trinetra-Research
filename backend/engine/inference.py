"""Thread-safe Apple Silicon MPS inference engine for KAAL-304M."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

import torch
from tokenizers import Tokenizer

from .config import (
    DEFAULT_DECODING_CONFIG,
    DEFAULT_MODEL_CONFIG,
    EOS_TOKEN,
    EXPECTED_UNIQUE_PARAMS,
    MODEL_NAME,
    DecodingConfig,
    ModelConfig,
    format_prompt,
)
from .model import KAALModel, count_unique_parameters


def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """Select Apple Silicon MPS (native FP32, ~1.22 GB) when available, otherwise CPU (FP32)."""
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: Sequence[int],
    penalty: float,
) -> torch.Tensor:
    """Apply sign-aware repetition penalty to previously generated token IDs."""
    if penalty == 1.0 or not generated_ids:
        return logits
    out = logits.clone()
    for token_id in set(generated_ids):
        score = out[token_id]
        if score < 0:
            out[token_id] = score * penalty
        else:
            out[token_id] = score / penalty
    return out


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: Sequence[int],
    ngram_size: int,
) -> torch.Tensor:
    """Hard-block any candidate token that would repeat an n-gram in generated_ids."""
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return logits

    prefix_len = ngram_size - 1
    if prefix_len == 0:
        banned = set(generated_ids)
    else:
        prefix = tuple(generated_ids[-prefix_len:])
        banned = set()
        limit = len(generated_ids) - ngram_size + 1
        for i in range(limit):
            if tuple(generated_ids[i : i + prefix_len]) == prefix:
                banned.add(generated_ids[i + prefix_len])

    if not banned:
        return logits

    out = logits.clone()
    for token_id in banned:
        out[token_id] = float("-inf")
    return out


def sample_next_token(
    logits_fp32: torch.Tensor,
    generated_ids: Sequence[int],
    decoding: DecodingConfig = DEFAULT_DECODING_CONFIG,
) -> int:
    """Apply generated-token penalties and sample from the filtered FP32 logits."""
    logits = logits_fp32.float()
    logits = apply_repetition_penalty(logits, generated_ids, decoding.repetition_penalty)
    if generated_ids and (decoding.frequency_penalty or decoding.presence_penalty):
        counts = Counter(generated_ids)
        ids = list(counts)
        penalties = logits.new_tensor([
            decoding.presence_penalty + decoding.frequency_penalty * counts[token] for token in ids
        ])
        logits = logits.clone()
        logits[ids] -= penalties
    logits = apply_no_repeat_ngram(logits, generated_ids, decoding.no_repeat_ngram_size)

    if not decoding.do_sample or decoding.temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / decoding.temperature

    vocab_size = logits.shape[-1]
    if decoding.top_k > 0 and decoding.top_k < vocab_size:
        top_vals, _ = torch.topk(logits, decoding.top_k)
        threshold = top_vals[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if decoding.min_p:
        threshold = logits.max() + math.log(decoding.min_p)
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < decoding.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > decoding.top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False  # Always retain at least the highest-probability token.
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return int(torch.argmax(logits_fp32).item())

    return int(torch.multinomial(probs, num_samples=1).item())


class KAALEngine:
    """Thread-safe local inference engine for KAAL-304M on Apple Silicon and CPU."""

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

            model.to(device=self.device, dtype=self.dtype)
            model.out_head.weight = model.tok_emb.weight
            model.eval()

            if model.out_head.weight.data_ptr() != model.tok_emb.weight.data_ptr():
                raise RuntimeError("Weight tying broken after moving KAALModel to device")

            unique_params = count_unique_parameters(model)
            if unique_params != EXPECTED_UNIQUE_PARAMS:
                raise RuntimeError(
                    f"Unexpected unique parameter count: {unique_params} != {EXPECTED_UNIQUE_PARAMS}"
                )

            # 1-token warmup pass to compile Metal shaders on Apple Silicon MPS.
            with torch.inference_mode():
                dummy = torch.tensor([[eos_id]], dtype=torch.long, device=self.device)
                _ = model(dummy, past_kv=None, use_cache=True)
                if self.device.type == "mps":
                    torch.mps.synchronize()

            self.tokenizer = tokenizer
            self.eos_id = int(eos_id)
            self.model = model
            self.unique_params = unique_params
            self.weights_loaded = True
            self.status = "ready"
            return True

    def stream(
        self,
        instruction: str,
        context: str = "",
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream UTF-8 BPE text deltas and final generation telemetry."""
        if not self.weights_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("KAAL-304M weights are not loaded.")

        decoding = self.decoding_config
        prompt = format_prompt(instruction, context)
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids: List[int] = list(encoded.ids)
        if not prompt_ids:
            prompt_ids = [self.eos_id]

        max_prompt_tokens = self.model_config.context_length - decoding.max_new_tokens
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        with self._lock:
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

                if self.device.type == "mps":
                    torch.mps.synchronize()

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
