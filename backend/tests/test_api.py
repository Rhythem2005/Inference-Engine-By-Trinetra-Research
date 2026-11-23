"""API tests for the KAAL-304M FastAPI backend."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import create_app


def test_health_reports_weights_missing_cleanly(tmp_path: Path) -> None:
    """Backend must start without crashing when weights are absent and report missing_files."""
    app = create_app(weights_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "weights_missing"
        assert data["model_name"] == "KAAL-304M"
        assert data["weights_loaded"] is False
        assert data["params"] == 304_355_328
        assert set(data["missing_files"]) == {"weights/best_model.pt", "weights/tokenizer.json"}


def test_hyperparameter_overrides_rejected_with_422(tmp_path: Path) -> None:
    """Requests attempting to pass decoding parameters must be rejected (extra='forbid')."""
    app = create_app(weights_dir=tmp_path)
    with TestClient(app) as client:
        for forbidden_key, value in [
            ("temperature", 0.9),
            ("top_k", 10),
            ("top_p", 0.5),
            ("repetition_penalty", 1.0),
            ("seed", 42),
            ("max_tokens", 256),
        ]:
            resp = client.post(
                "/api/chat/stream",
                json={"message": "Hello", forbidden_key: value},
            )
            assert resp.status_code == 422, f"{forbidden_key} was not rejected"

            resp_sync = client.post(
                "/api/chat",
                json={"message": "Hello", forbidden_key: value},
            )
            assert resp_sync.status_code == 422, f"{forbidden_key} was not rejected on /api/chat"


def test_stream_returns_sse_error_when_weights_missing(tmp_path: Path) -> None:
    """/api/chat/stream should emit an SSE error payload when weights are missing."""
    app = create_app(weights_dir=tmp_path)
    with TestClient(app) as client:
        with client.stream("POST", "/api/chat/stream", json={"message": "Hi"}) as resp:
            assert resp.status_code == 200
            lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
            assert len(lines) == 1
            payload = json.loads(lines[0].removeprefix("data: "))
            assert payload["type"] == "error"
            assert "weights" in payload["message"].lower()


def test_cors_allows_vite_dev_origins(tmp_path: Path) -> None:
    """CORS headers must permit localhost:5173 and 127.0.0.1:5173."""
    app = create_app(weights_dir=tmp_path)
    with TestClient(app) as client:
        for origin in ["http://localhost:5173", "http://127.0.0.1:5173"]:
            resp = client.options(
                "/api/chat/stream",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert resp.status_code == 200
            assert resp.headers.get("access-control-allow-origin") == origin
