"""FastAPI surface for the LangGraph policy Q&A POC."""

import asyncio
import json
import os
import secrets
import time
import uuid
from collections import defaultdict

if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(logger_name="policy-qa")

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .graph import build_graph, qa_invocations, valid_citations
from .retriever import load_documents

app = FastAPI(title="langgraph-policy-qa-poc", version="0.4.0")
graph = build_graph()

_STARTED_AT = time.time()

# ---------------------------------------------------------------------------
# In-memory sliding-window rate limiter (per API key, 30 req / 60s)
# Thread-safe via asyncio.Lock.
# ---------------------------------------------------------------------------
_RATE_LIMIT = int(os.environ.get("RATE_LIMIT_RPM", "30"))
_RATE_WINDOW = 60.0
_PRUNE_INTERVAL = 300.0  # prune stale keys every 5 min
_request_log: dict[str, list[float]] = defaultdict(list)
_rate_lock = asyncio.Lock()
_last_prune = time.time()


async def _check_rate_limit(api_key: str) -> None:
    global _last_prune
    async with _rate_lock:
        now = time.time()
        # Periodic pruning of stale keys
        if now - _last_prune > _PRUNE_INTERVAL:
            stale = [k for k, v in _request_log.items()
                     if not v or now - v[-1] > _RATE_WINDOW]
            for k in stale:
                del _request_log[k]
            _last_prune = now
        window = _request_log[api_key]
        # Evict expired entries
        _request_log[api_key] = window = [t for t in window if now - t < _RATE_WINDOW]
        if len(window) >= _RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({_RATE_LIMIT} requests per minute)",
            )
        window.append(now)


class QARequest(BaseModel):
    question: str = Field(..., max_length=2000)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return v.strip()


async def verify_api_key(request: Request) -> None:
    expected = os.environ.get("POC_API_KEY", "")
    provided = request.headers.get("x-api-key", "")
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    await _check_rate_limit(provided)


def _format_result(thread_id: str, result: dict, latency_ms: int, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "thread_id": thread_id,
        "intent": result.get("intent"),
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "grounded": result.get("grounded", False),
        "retrieved_docs": sorted({c["doc"] for c in result.get("retrieved", [])}),
        "latency_ms": latency_ms,
    }


def _record_metrics(result: dict) -> None:
    intent = result.get("intent") or "unknown"
    grounded = bool(result.get("grounded", False))
    qa_invocations.add(1, {"intent": intent, "grounded": str(grounded)})
    valid_citations.add(len(result.get("citations", [])), {"intent": intent})


@app.get("/")
async def root() -> dict:
    return {
        "service": "langgraph-policy-qa-poc",
        "version": "0.4.0",
        "endpoints": {
            "health": "/healthz",
            "qa": "POST /v1/qa/{thread_id}",
            "qa_stream": "POST /v1/qa/{thread_id}/stream",
        },
    }


@app.get("/healthz")
async def healthz(request: Request) -> dict:
    uptime_s = round(time.time() - _STARTED_AT)
    docs = load_documents()

    # Minimal response for unauthenticated callers
    expected = os.environ.get("POC_API_KEY", "")
    provided = request.headers.get("x-api-key", "")
    authenticated = bool(expected and secrets.compare_digest(provided, expected))

    if not authenticated:
        return {"status": "ok", "uptime_seconds": uptime_s}

    # Full diagnostics for authenticated callers
    llm_ok = False
    llm_latency_ms = 0
    try:
        from . import llm
        t0 = time.time()
        result = await asyncio.to_thread(
            llm.chat_json,
            "Reply with json: {\"status\": \"ok\"}",
            "health check",
        )
        llm_latency_ms = round((time.time() - t0) * 1000)
        llm_ok = result.get("status") == "ok"
    except Exception:
        pass

    status = "ok" if llm_ok else "degraded"
    return {
        "status": status,
        "uptime_seconds": uptime_s,
        "documents": len(docs),
        "llm": {"reachable": llm_ok, "latency_ms": llm_latency_ms},
    }


@app.post("/v1/qa/{thread_id}", dependencies=[Depends(verify_api_key)])
async def qa(thread_id: str, body: QARequest) -> dict:
    request_id = uuid.uuid4().hex
    started = time.time()
    try:
        result = await asyncio.to_thread(
            graph.invoke,
            {"question": body.question},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="graph invocation failed") from exc
    _record_metrics(result)
    return _format_result(thread_id, result, round((time.time() - started) * 1000), request_id)


@app.post("/v1/qa/{thread_id}/stream", dependencies=[Depends(verify_api_key)])
async def qa_stream(thread_id: str, body: QARequest) -> StreamingResponse:
    """Server-Sent Events endpoint — streams node completions as they happen."""

    request_id = uuid.uuid4().hex

    async def event_generator():
        started = time.time()
        last_state: dict = {}
        try:
            async for event in graph.astream(
                {"question": body.question},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    last_state.update(node_output)
                    payload = json.dumps({"node": node_name, "data": node_output})
                    yield f"event: node\ndata: {payload}\n\n"
        except Exception as exc:
            err = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {err}\n\n"
            return

        _record_metrics(last_state)
        final = _format_result(
            thread_id, last_state, round((time.time() - started) * 1000), request_id,
        )
        yield f"event: done\ndata: {json.dumps(final)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
