"""Unit tests for the KAAL-304M Apple Silicon inference engine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from backend.engine.config import (
    DEFAULT_MODEL_CONFIG,
    EXPECTED_UNIQUE_PARAMS,
    LOCKED_DECODING_CONFIG,
    ModelConfig,
    format_prompt,
)
from backend.engine.inference import (
    KAALEngine,
    apply_no_repeat_ngram,
    apply_repetition_penalty,
)
from backend.engine.model import KAALModel, count_unique_parameters


def test_exact_unique_parameter_count_and_tied_weights() -> None:
    """KAAL-304M must have exactly 304,355,328 unique parameters and tied weights."""
    model = KAALModel(DEFAULT_MODEL_CONFIG)
    assert model.out_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    assert count_unique_parameters(model) == EXPECTED_UNIQUE_PARAMS == 304_355_328


def test_kv_cache_logit_parity() -> None:
    """Prefill + cached single-token decode must match full-sequence logits within <1e-4."""
    torch.manual_seed(42)
    small_cfg = ModelConfig(
        vocab_size=256,
        context_length=64,
        emb_dim=64,
        n_layers=3,
        n_heads=4,
        ffn_dim=256,
        drop_rate=0.0,
        qkv_bias=True,
    )
    model = KAALModel(small_cfg).eval()
    tokens = torch.randint(0, small_cfg.vocab_size, (1, 16), dtype=torch.long)

    with torch.inference_mode():
        full_logits = model(tokens, past_kv=None, use_cache=False)

        # Prefill first 10 tokens, then decode remaining 6 tokens one-by-one with KV-cache.
        prefill_logits, past_kv = model(tokens[:, :10], past_kv=None, use_cache=True)
        cached_steps = [prefill_logits]
        for step in range(10, 16):
            step_logits, past_kv = model(tokens[:, step : step + 1], past_kv=past_kv, use_cache=True)
            cached_steps.append(step_logits)
        cached_logits = torch.cat(cached_steps, dim=1)

    max_diff = (full_logits - cached_logits).abs().max().item()
    assert max_diff < 1e-4, f"KV-cache logit mismatch: max_diff={max_diff}"


def test_canonical_prompt_format() -> None:
    """Prompt formatting must strictly preserve trailing newline after ### Response:."""
    p1 = format_prompt("  What is the capital of France?  ")
    assert p1 == "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
    assert p1.endswith("### Response:\n")

    p2 = format_prompt("Summarize this.", "  KAAL-304M runs locally on Apple Silicon.  ")
    assert p2 == (
        "### Instruction:\nSummarize this.\n\n"
        "### Input:\nKAAL-304M runs locally on Apple Silicon.\n\n"
        "### Response:\n"
    )
    assert p2.endswith("### Response:\n")


def test_locked_decoding_config_is_immutable() -> None:
    """Locked decoding hyperparameters must match spec and reject mutation."""
    assert LOCKED_DECODING_CONFIG.do_sample is True
    assert LOCKED_DECODING_CONFIG.temperature == 0.45
    assert LOCKED_DECODING_CONFIG.top_k == 35
    assert LOCKED_DECODING_CONFIG.top_p == 0.90
    assert LOCKED_DECODING_CONFIG.repetition_penalty == 1.18
    assert LOCKED_DECODING_CONFIG.no_repeat_ngram_size == 3
    assert LOCKED_DECODING_CONFIG.max_new_tokens == 128

    with pytest.raises(FrozenInstanceError):
        LOCKED_DECODING_CONFIG.temperature = 0.9  # type: ignore[misc]


def test_sign_aware_repetition_penalty_and_ngram_blocker() -> None:
    """Repetition penalty must be sign-aware and 3-gram blocker must mask repeated trigrams."""
    logits = torch.tensor([2.0, -2.0, 1.0, 0.5], dtype=torch.float32)
    penalized = apply_repetition_penalty(logits, [0, 1], penalty=2.0)
    assert torch.isclose(penalized[0], torch.tensor(1.0))
    assert torch.isclose(penalized[1], torch.tensor(-4.0))
    assert torch.isclose(penalized[2], torch.tensor(1.0))

    # Sequence [1, 2, 3, 1, 2] -> prefix (1, 2) was previously followed by 3 -> token 3 must be -inf.
    blocked = apply_no_repeat_ngram(logits, [1, 2, 3, 1, 2], ngram_size=3)
    assert blocked[3].item() == float("-inf")
    assert blocked[0].item() == 2.0


def test_missing_weights_graceful_state(tmp_path: Path) -> None:
    """KAALEngine must report weights_missing cleanly without crashing when files are absent."""
    engine = KAALEngine(weights_dir=tmp_path)
    loaded = engine.load()
    assert loaded is False
    health = engine.health_status()
    assert health["status"] == "weights_missing"
    assert health["weights_loaded"] is False
    assert set(health["missing_files"]) == {"weights/best_model.pt", "weights/tokenizer.json"}


def test_real_checkpoint_strict_load_when_present() -> None:
    """If weights/best_model.pt and weights/tokenizer.json exist, verify strict=True load and tied weights."""
    repo_root = Path(__file__).resolve().parents[2]
    weights_dir = repo_root / "weights"
    if not (weights_dir / "best_model.pt").is_file() or not (weights_dir / "tokenizer.json").is_file():
        pytest.skip("Local weights not present in ./weights/")

    engine = KAALEngine(weights_dir=weights_dir)
    assert engine.load() is True
    assert engine.model is not None
    assert engine.eos_id == 0
    assert engine.model.out_head.weight.data_ptr() == engine.model.tok_emb.weight.data_ptr()
    assert count_unique_parameters(engine.model) == EXPECTED_UNIQUE_PARAMS
