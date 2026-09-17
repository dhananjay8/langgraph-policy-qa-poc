"""FastAPI surface for the LangGraph policy Q&A POC."""

import os
import secrets
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from .graph import build_graph
from .retriever import load_documents

app = FastAPI(title="langgraph-policy-qa-poc", version="0.1.0")
graph = build_graph()


class QARequest(BaseModel):
    question: str


def verify_api_key(request: Request) -> None:
    expected = os.environ.get("POC_API_KEY", "")
    provided = request.headers.get("x-api-key", "")
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/")
def root() -> dict:
    return {
        "service": "langgraph-policy-qa-poc",
        "version": "0.1.1",
        "endpoints": {"health": "/healthz", "qa": "POST /v1/qa/{thread_id}"},
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "documents": len(load_documents())}


@app.post("/v1/qa/{thread_id}", dependencies=[Depends(verify_api_key)])
def qa(thread_id: str, body: QARequest) -> dict:
    started = time.time()
    try:
        result = graph.invoke(
            {"question": body.question},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="graph invocation failed") from exc
    return {
        "thread_id": thread_id,
        "intent": result.get("intent"),
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "grounded": result.get("grounded", False),
        "retrieved_docs": sorted({c["doc"] for c in result.get("retrieved", [])}),
        "latency_ms": round((time.time() - started) * 1000),
    }
