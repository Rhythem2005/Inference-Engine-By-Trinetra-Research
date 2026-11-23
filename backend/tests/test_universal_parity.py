"""Universal input-agnostic mathematical parity suite for KAAL-304M.

Verifies exact parity between KAALEngine (MPS/CPU + KV-cache) and uncached
CPU FP32 ground truth across:
  1. 10 arbitrary random token sequences spanning L in {1, 3, 4, 5, 7, 16, 33, 64, 128, 512}
     with 20 autoregressive KV-cached decode steps each.
  2. Context-window, prompt-variant, and back-to-back cache-reset parity.
  3. Byte-exact incremental UTF-8 streaming decoder parity.
  4. 4-stage diagnostic table comparing CPU FP32, MPS FP32, MPS FP16, and KAALEngine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
from tokenizers import Tokenizer

from backend.engine import (
    DEFAULT_MODEL_CONFIG,
    EXPECTED_UNIQUE_PARAMS,
    KAALEngine,
    KAALModel,
    count_unique_parameters,
    format_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "weights"
RANDOM_LENGTHS = (1, 3, 4, 5, 7, 16, 33, 64, 128, 512)
DECODE_STEPS = 20


def _require_weights() -> None:
    if not (WEIGHTS_DIR / "best_model.pt").is_file() or not (WEIGHTS_DIR / "tokenizer.json").is_file():
        pytest.skip("Real weights not present in ./weights/")


@pytest.fixture(scope="module")
def parity_context() -> Tuple[KAALEngine, KAALModel, Tokenizer]:
    """Load KAALEngine once and a separate uncached CPU FP32 reference model."""
    _require_weights()

    engine = KAALEngine(weights_dir=WEIGHTS_DIR)
    assert engine.load() is True
    assert engine.model is not None
    assert engine.tokenizer is not None
    assert engine.eos_id == 0
    assert engine.model.out_head.weight.data_ptr() == engine.model.tok_emb.weight.data_ptr()
    assert count_unique_parameters(engine.model) == EXPECTED_UNIQUE_PARAMS

    payload = torch.load(WEIGHTS_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    ref_model = KAALModel(DEFAULT_MODEL_CONFIG).eval()
    ref_model.load_state_dict(state_dict, strict=True)
    ref_model.to(device=torch.device("cpu"), dtype=torch.float32)
    ref_model.out_head.weight = ref_model.tok_emb.weight

    return engine, ref_model, engine.tokenizer


def _compare_prefill_and_autoregressive_decode(
    engine: KAALEngine,
    ref_model: KAALModel,
    prompt_ids: List[int],
    decode_steps: int,
) -> Tuple[float, bool]:
    """Compare KAALEngine (KV-cached) vs CPU FP32 (uncached) at prefill + every decode step."""
    assert engine.model is not None
    dev = engine.device
    max_abs_diff = 0.0
    all_argmax_match = True

    with torch.inference_mode():
        # Prefill step
        ref_seq = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
        eng_in = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

        ref_logits_all = ref_model(ref_seq, past_kv=None, use_cache=False)
        eng_logits_all, past_kv = engine.model(eng_in, past_kv=None, use_cache=True)

        ref_last = ref_logits_all[0, -1, :].float()
        eng_last = eng_logits_all[0, -1, :].float().cpu()

        step_diff = (ref_last - eng_last).abs().max().item()
        max_abs_diff = max(max_abs_diff, step_diff)
        ref_tok = int(ref_last.argmax().item())
        eng_tok = int(eng_last.argmax().item())
        if ref_tok != eng_tok:
            all_argmax_match = False

        # Autoregressive single-token decode steps (strictly greedy next token from reference)
        next_tok = ref_tok
        for _ in range(decode_steps):
            ref_seq = torch.cat(
                [ref_seq, torch.tensor([[next_tok]], dtype=torch.long, device="cpu")], dim=1
            )
            eng_step = torch.tensor([[next_tok]], dtype=torch.long, device=dev)

            ref_logits_all = ref_model(ref_seq, past_kv=None, use_cache=False)
            eng_logits_step, past_kv = engine.model(eng_step, past_kv=past_kv, use_cache=True)

            ref_last = ref_logits_all[0, -1, :].float()
            eng_last = eng_logits_step[0, -1, :].float().cpu()

            step_diff = (ref_last - eng_last).abs().max().item()
            max_abs_diff = max(max_abs_diff, step_diff)
            ref_tok = int(ref_last.argmax().item())
            eng_tok = int(eng_last.argmax().item())
            if ref_tok != eng_tok:
                all_argmax_match = False
            next_tok = ref_tok

    return max_abs_diff, all_argmax_match


@pytest.mark.parametrize("seq_len", RANDOM_LENGTHS)
def test_random_token_sequence_universal_parity(
    parity_context: Tuple[KAALEngine, KAALModel, Tokenizer],
    seq_len: int,
) -> None:
    """Prefill + 20 cached decode steps must match uncached CPU FP32 across all lengths L."""
    engine, ref_model, _ = parity_context
    gen = torch.Generator().manual_seed(1000 + seq_len)
    prompt_ids = torch.randint(1, DEFAULT_MODEL_CONFIG.vocab_size, (seq_len,), generator=gen).tolist()

    max_diff, argmax_match = _compare_prefill_and_autoregressive_decode(
        engine=engine,
        ref_model=ref_model,
        prompt_ids=prompt_ids,
        decode_steps=DECODE_STEPS,
    )
    # Across 20 layers x 49,152 logits, FP32 GEMV (M=1) vs GEMM (M>1) reduction order
    # introduces ~1e-4 noise; 5e-4 stays 2x tighter than 1e-3 and ~50x tighter than FP16 drift.
    tol = 5e-4 if engine.dtype == torch.float32 else 1e-3
    assert max_diff < tol, f"L={seq_len}: max_abs_diff={max_diff:.6e} >= {tol}"
    assert argmax_match, f"L={seq_len}: argmax token mismatch against CPU FP32"


def test_context_variants_and_zero_cache_leakage(
    parity_context: Tuple[KAALEngine, KAALModel, Tokenizer],
) -> None:
    """Verify prompt variants (empty/long context, casing) and zero KV-cache leakage across calls."""
    engine, ref_model, tokenizer = parity_context
    long_context = (
        "Apple Silicon unifies CPU and GPU memory into a single high-bandwidth pool. "
        "PyTorch MPS executes decoder-only Transformer layers directly on the GPU. "
    ) * 6

    variants = [
        ("What is the capital of France?", ""),
        ("WHAT IS THE CAPITAL OF FRANCE?", ""),
        ("what is the capital of france?", ""),
        ("Summarize the key architectural benefit.", long_context),
        ("List three numbers: 1, 2, 3.", "Numbers: 10, 20, 30.\nLine two: 40, 50, 60."),
    ]

    # Run back-to-back and repeat first prompt at the end to prove zero cross-request leakage.
    variants.append(variants[0])

    first_logits_snapshot = None
    for idx, (instruction, ctx) in enumerate(variants):
        prompt = format_prompt(instruction, ctx)
        assert prompt.endswith("### Response:\n")
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids

        max_diff, argmax_match = _compare_prefill_and_autoregressive_decode(
            engine=engine,
            ref_model=ref_model,
            prompt_ids=ids,
            decode_steps=5,
        )
        assert max_diff < 1e-4, f"Variant {idx} max_abs_diff={max_diff:.6e}"
        assert argmax_match, f"Variant {idx} argmax mismatch"

        if idx == 0 or idx == len(variants) - 1:
            with torch.inference_mode():
                assert engine.model is not None
                out, _ = engine.model(
                    torch.tensor([ids], dtype=torch.long, device=engine.device),
                    past_kv=None,
                    use_cache=True,
                )
                snap = out[0, -1, :].float().cpu()
            if idx == 0:
                first_logits_snapshot = snap
            else:
                assert first_logits_snapshot is not None
                assert torch.equal(first_logits_snapshot, snap), "Cross-request KV-cache leakage detected"


def test_byte_exact_streaming_decoder(
    parity_context: Tuple[KAALEngine, KAALModel, Tokenizer],
) -> None:
    """Concatenated streamed deltas must equal tokenizer.decode(generated_ids) byte-for-byte."""
    engine, _, tokenizer = parity_context

    prompts = [
        ("List three ways to save electricity at home.", ""),
        ("Count from 1 to 5 with punctuation and newlines.", ""),
        (
            "Summarize this paragraph.",
            "In 2026, local 304M models run on laptops with 100% privacy, fast token/s, and zero API fees.",
        ),
    ]

    for seed, (instruction, ctx) in enumerate(prompts):
        # Capture streamed deltas and generated token IDs by running with identical RNG seed.
        torch.manual_seed(100 + seed)
        streamed_chunks: List[str] = []
        done_event: Dict[str, object] = {}
        for ev in engine.stream(instruction, ctx):
            if ev["type"] == "token":
                streamed_chunks.append(ev["text"])
            elif ev["type"] == "done":
                done_event = ev

        streamed_text = "".join(streamed_chunks)
        assert int(done_event["tokens"]) > 0

        # Re-run exact same seed directly through model+sampler to collect exact generated_ids.
        from backend.engine.inference import sample_next_token

        torch.manual_seed(100 + seed)
        prompt = format_prompt(instruction, ctx)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        gen_ids: List[int] = []
        assert engine.model is not None
        with torch.inference_mode():
            logits, past_kv = engine.model(
                torch.tensor([prompt_ids], dtype=torch.long, device=engine.device),
                past_kv=None,
                use_cache=True,
            )
            next_logits = logits[0, -1, :].float()
            for _ in range(engine.decoding_config.max_new_tokens):
                tok = sample_next_token(next_logits, gen_ids, engine.decoding_config)
                if tok == engine.eos_id:
                    break
                gen_ids.append(tok)
                if len(gen_ids) >= engine.decoding_config.max_new_tokens:
                    break
                logits, past_kv = engine.model(
                    torch.tensor([[tok]], dtype=torch.long, device=engine.device),
                    past_kv=past_kv,
                    use_cache=True,
                )
                next_logits = logits[0, -1, :].float()

        expected_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        assert streamed_text.encode("utf-8") == expected_text.encode("utf-8"), (
            f"Byte mismatch!\nstreamed: {streamed_text!r}\nexpected: {expected_text!r}"
        )


def test_four_stage_diagnostic_table(
    parity_context: Tuple[KAALEngine, KAALModel, Tokenizer],
) -> None:
    """Print 4-stage diagnostic max_abs_diff table vs CPU FP32 uncached ground truth."""
    engine, ref_model, tokenizer = parity_context
    prompt = format_prompt("What is the capital of France?")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    x_cpu = torch.tensor([ids], dtype=torch.long, device="cpu")

    with torch.inference_mode():
        ref_logits = ref_model(x_cpu, past_kv=None, use_cache=False)[0, -1, :].float()

        # (A) CPU FP32 uncached
        diff_a = (ref_logits - ref_logits).abs().max().item()

        # (B) MPS FP32 uncached & (C) MPS FP16 uncached (or CPU FP16 diagnostic if MPS unavailable)
        mps_avail = torch.backends.mps.is_available()
        if mps_avail:
            m_mps32 = KAALModel(DEFAULT_MODEL_CONFIG).eval()
            m_mps32.load_state_dict(ref_model.state_dict(), strict=True)
            m_mps32.to(device=torch.device("mps"), dtype=torch.float32)
            m_mps32.out_head.weight = m_mps32.tok_emb.weight
            out_b = m_mps32(x_cpu.to("mps"), past_kv=None, use_cache=False)[0, -1, :].float().cpu()
            diff_b_str = f"{(ref_logits - out_b).abs().max().item():.6e}"
            del m_mps32

            m_mps16 = KAALModel(DEFAULT_MODEL_CONFIG).eval()
            m_mps16.load_state_dict(ref_model.state_dict(), strict=True)
            m_mps16.to(device=torch.device("mps"), dtype=torch.float16)
            m_mps16.out_head.weight = m_mps16.tok_emb.weight
            out_c = m_mps16(x_cpu.to("mps"), past_kv=None, use_cache=False)[0, -1, :].float().cpu()
            diff_c_str = f"{(ref_logits - out_c).abs().max().item():.6e}"
            del m_mps16
        else:
            diff_b_str = "N/A (MPS unavailable in sandbox)"
            m_cpu16 = KAALModel(DEFAULT_MODEL_CONFIG).eval()
            m_cpu16.load_state_dict(ref_model.state_dict(), strict=True)
            m_cpu16.to(device=torch.device("cpu"), dtype=torch.float16)
            m_cpu16.out_head.weight = m_cpu16.tok_emb.weight
            out_c = m_cpu16(x_cpu, past_kv=None, use_cache=False)[0, -1, :].float()
            diff_c_str = f"{(ref_logits - out_c).abs().max().item():.6e} (CPU FP16 fallback)"
            del m_cpu16

        # (D) KAALEngine (device + KV-cache)
        assert engine.model is not None
        out_d, _ = engine.model(x_cpu.to(engine.device), past_kv=None, use_cache=True)
        diff_d = (ref_logits - out_d[0, -1, :].float().cpu()).abs().max().item()

    print("\n=== 4-Stage Diagnostic Table (max_abs_diff vs CPU FP32 uncached) ===")
    print(f"(A) CPU FP32 uncached             : {diff_a:.6e}")
    print(f"(B) MPS FP32 uncached             : {diff_b_str}")
    print(f"(C) MPS FP16 uncached             : {diff_c_str}")
    print(f"(D) KAALEngine ({engine.device} {str(engine.dtype).replace('torch.', '')} + KV) : {diff_d:.6e}")

    assert diff_d < 1e-4, f"KAALEngine drift too high: {diff_d:.6e}"
