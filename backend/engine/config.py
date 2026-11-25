"""Immutable configuration and canonical prompt formatting for KAAL-304M."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

MODEL_NAME = "KAAL-304M"
EXPECTED_UNIQUE_PARAMS = 304_355_328
EOS_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class ModelConfig:
    """Exact KAAL-304M architecture configuration."""

    vocab_size: int = 49_152
    context_length: int = 2_048
    emb_dim: int = 1_024
    n_layers: int = 20
    n_heads: int = 16
    ffn_dim: int = 4_096
    drop_rate: float = 0.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.n_heads


@dataclass(frozen=True)
class DecodingConfig:
    """Immutable settings for a KAAL-304M generation profile."""

    do_sample: bool = False
    temperature: float = 0.45
    top_k: int = 35
    top_p: float = 0.90
    repetition_penalty: float = 1.18
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 256
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self) -> None:
        if type(self.do_sample) is not bool:
            raise ValueError("do_sample must be a boolean")
        for name in ("top_k", "no_repeat_ngram_size", "max_new_tokens"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.top_k < 0 or self.no_repeat_ngram_size < 0 or self.max_new_tokens < 1:
            raise ValueError("top_k and no_repeat_ngram_size must be nonnegative and max_new_tokens positive")
        for name in ("temperature", "top_p", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.temperature < 0 or self.repetition_penalty <= 0:
            raise ValueError("temperature must be nonnegative and repetition_penalty positive")
        if not (0 < self.top_p <= 1 and 0 <= self.min_p <= 1):
            raise ValueError("top_p must be in (0, 1] and min_p in [0, 1]")


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_DECODING_CONFIG = DecodingConfig()
# Existing library callers can continue importing this name.
LOCKED_DECODING_CONFIG = DEFAULT_DECODING_CONFIG


def format_prompt(instruction: str, context: str = "") -> str:
    """Format user instruction and optional reference context into KAAL-304M prompt."""
    out = "### Instruction:\n" + instruction.strip() + "\n\n"
    ctx = (context or "").strip()
    if ctx:
        out += "### Input:\n" + ctx + "\n\n"
    return out + "### Response:\n"
