"""Exercise Windows device selection and CUDA fallback without requiring a GPU."""

from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from backend.windows import engine as windows


def test_auto_selects_nvidia_cuda_and_cpu_override_skips_detection(monkeypatch):
    monkeypatch.delenv("KAAL_DEVICE", raising=False)
    monkeypatch.setattr(torch.version, "cuda", "13.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert windows.select_device_and_dtype() == (torch.device("cuda:0"), torch.float32)

    def unexpected_detection():
        pytest.fail("CPU override should bypass CUDA detection")

    monkeypatch.setenv("KAAL_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", unexpected_detection)
    assert windows.select_device_and_dtype() == (torch.device("cpu"), torch.float32)


def test_cuda_initialization_failure_selects_cpu(monkeypatch):
    def unavailable_driver():
        raise RuntimeError("Simulated CUDA driver error")

    monkeypatch.setenv("KAAL_DEVICE", "auto")
    monkeypatch.setattr(torch.version, "cuda", "13.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", unavailable_driver)
    assert windows.select_device_and_dtype() == (torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("error", [torch.cuda.OutOfMemoryError, RuntimeError])
def test_cuda_setup_failure_loads_checkpoint_on_cpu(monkeypatch, tmp_path, error):
    config = replace(
        windows.DEFAULT_MODEL_CONFIG, vocab_size=2, n_layers=1, n_heads=1, emb_dim=8, ffn_dim=16
    )
    model = windows.KAALModel(config)
    torch.save({"model": model.state_dict()}, tmp_path / "best_model.pt")
    Tokenizer(WordLevel({windows.EOS_TOKEN: 0, "hello": 1})).save(str(tmp_path / "tokenizer.json"))
    monkeypatch.setattr(windows, "EXPECTED_UNIQUE_PARAMS", windows.count_unique_parameters(model))
    monkeypatch.setenv("KAAL_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    engine = windows.KAALEngine(weights_dir=tmp_path, model_config=config)
    engine.device = torch.device("cuda:0")
    prepare_model = engine._prepare_model

    def fail_cuda_setup(model, device, eos_id):
        if device.type == "cuda":
            raise error("Simulated CUDA setup error")
        return prepare_model(model, device, eos_id)

    monkeypatch.setattr(engine, "_prepare_model", fail_cuda_setup)
    assert engine.load()
    assert engine.health_status()["device"] == "cpu"
    assert engine.health_status()["status"] == "ready"
    assert engine.model.tok_emb.weight is engine.model.out_head.weight
    torch.testing.assert_close(engine.model.tok_emb.weight, model.tok_emb.weight)


@pytest.mark.parametrize("started_streaming", [False, True])
def test_cuda_generation_oom_switches_to_cpu(monkeypatch, started_streaming):
    monkeypatch.setenv("KAAL_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    engine = windows.KAALEngine()
    engine.model = windows.KAALModel(
        replace(windows.DEFAULT_MODEL_CONFIG, vocab_size=2, n_layers=1, n_heads=1, emb_dim=8, ffn_dim=16)
    )
    engine.device = torch.device("cuda:0")
    engine.tokenizer = Tokenizer(WordLevel({windows.EOS_TOKEN: 0, "hello": 1}))
    engine.weights_loaded = True

    def stream(instruction, context, cancel_event):
        if engine.device.type == "cuda":
            if started_streaming:
                yield {"type": "token", "text": "partial"}
            raise torch.cuda.OutOfMemoryError("Simulated CUDA generation error")
        yield {"type": "token", "text": "complete"}

    monkeypatch.setattr(engine, "_stream", stream)
    events = engine.stream("hello")
    if started_streaming:
        assert next(events)["text"] == "partial"
        with pytest.raises(windows.RetryOnCPU, match="switched to CPU"):
            next(events)
        events = engine.stream("hello")
    assert list(events) == [{"type": "token", "text": "complete"}]
    assert engine.device.type == "cpu"
