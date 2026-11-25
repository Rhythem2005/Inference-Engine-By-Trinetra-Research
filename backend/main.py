"""FastAPI backend for KAAL-304M Local with SSE streaming."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.engine import DEFAULT_DECODING_CONFIG, DecodingConfig, KAALEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights"
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


class ChatRequest(BaseModel):
    """Chat requests use the server's generation profile."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=16_000)
    context: Optional[str] = Field(default="", max_length=32_000)

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty")
        return value


class ChatResponse(BaseModel):
    """Non-streaming chat response payload."""

    text: str
    tokens: int
    tokens_per_second: float
    stopped_reason: str


def _format_sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _configured_decoding() -> DecodingConfig:
    raw = os.environ.get("KAAL_DECODING")
    if raw is None:
        return DEFAULT_DECODING_CONFIG
    try:
        settings = json.loads(raw)
        if not isinstance(settings, dict):
            raise ValueError("expected a JSON object")
        return replace(DEFAULT_DECODING_CONFIG, **settings)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid KAAL_DECODING: {exc}") from exc


def create_app(
    weights_dir: Optional[str | Path] = None,
    *,
    decoding_config: DecodingConfig | None = None,
) -> FastAPI:
    """Create and configure the KAAL-304M FastAPI application."""
    resolved_weights = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    engine = KAALEngine(
        weights_dir=resolved_weights, decoding_config=decoding_config or _configured_decoding()
    )
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await asyncio.to_thread(engine.load)
        yield

    app = FastAPI(
        title="KAAL-304M Local API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.inference_lock = inference_lock

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return engine.health_status()

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        if not engine.weights_loaded:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "KAAL-304M weights are missing or not loaded.",
                    "missing_files": engine.missing_files(),
                },
            )

        async with inference_lock:
            result = await asyncio.to_thread(
                engine.generate,
                payload.message,
                payload.context or "",
            )
        return ChatResponse(**result)

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        async def event_generator() -> AsyncGenerator[str, None]:
            if not engine.weights_loaded:
                yield _format_sse(
                    {
                        "type": "error",
                        "message": "KAAL-304M weights are missing. Place best_model.pt and tokenizer.json in ./weights/ and restart.",
                    }
                )
                return

            cancel_event = threading.Event()
            queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            async with inference_lock:
                if await request.is_disconnected():
                    return

                def producer() -> None:
                    try:
                        for event in engine.stream(
                            instruction=payload.message,
                            context=payload.context or "",
                            cancel_event=cancel_event,
                        ):
                            loop.call_soon_threadsafe(queue.put_nowait, event)
                    except Exception:
                        logger.exception("KAAL-304M streaming generation failed")
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"type": "error", "message": "Generation failed. Check the server log."},
                        )
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                worker = threading.Thread(target=producer, daemon=True)
                worker.start()

                try:
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            break
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue
                        if item is None:
                            break
                        yield _format_sse(item)
                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                finally:
                    cancel_event.set()
                    await asyncio.to_thread(worker.join)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
