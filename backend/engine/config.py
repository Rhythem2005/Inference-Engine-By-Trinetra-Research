"""Immutable configuration and canonical prompt formatting for KAAL-304M."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Strictly locked decoding hyperparameters for KAAL-304M."""

    do_sample: bool = True
    temperature: float = 0.45
    top_k: int = 35
    top_p: float = 0.90
    repetition_penalty: float = 1.18
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 256


DEFAULT_MODEL_CONFIG = ModelConfig()
LOCKED_DECODING_CONFIG = DecodingConfig()


def format_prompt(instruction: str, context: str = "") -> str:
    """Format user instruction and optional reference context into KAAL-304M prompt."""
    out = "### Instruction:\n" + instruction.strip() + "\n\n"
    ctx = (context or "").strip()
    if ctx:
        out += "### Input:\n" + ctx + "\n\n"
    return out + "### Response:\n"
