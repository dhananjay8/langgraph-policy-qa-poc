"""FastAPI surface for the LangGraph policy Q&A POC."""

import asyncio
import json
import os
import secrets
import time

if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(logger_name="policy-qa")

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .graph import build_graph, qa_invocations, valid_citations
from .retriever import load_documents

app = FastAPI(title="langgraph-policy-qa-poc", version="0.2.0")
graph = build_graph()


class QARequest(BaseModel):
    question: str = Field(..., max_length=2000)


def verify_api_key(request: Request) -> None:
    expected = os.environ.get("POC_API_KEY", "")
    provided = request.headers.get("x-api-key", "")
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _format_result(thread_id: str, result: dict, latency_ms: int) -> dict:
    return {
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
        "version": "0.2.0",
        "endpoints": {
            "health": "/healthz",
            "qa": "POST /v1/qa/{thread_id}",
            "qa_stream": "POST /v1/qa/{thread_id}/stream",
        },
    }


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "documents": len(load_documents())}


@app.post("/v1/qa/{thread_id}", dependencies=[Depends(verify_api_key)])
async def qa(thread_id: str, body: QARequest) -> dict:
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
    return _format_result(thread_id, result, round((time.time() - started) * 1000))


@app.post("/v1/qa/{thread_id}/stream", dependencies=[Depends(verify_api_key)])
async def qa_stream(thread_id: str, body: QARequest) -> StreamingResponse:
    """Server-Sent Events endpoint — streams node completions as they happen."""

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
            thread_id, last_state, round((time.time() - started) * 1000),
        )
        yield f"event: done\ndata: {json.dumps(final)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
