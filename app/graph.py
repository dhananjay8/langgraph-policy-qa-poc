"""LangGraph policy Q&A graph.

Flow:
    guard -> classify -> (policy_question | off_topic)
    policy_question -> retrieve -> rerank -> generate -> validate ─┐
                                                ▲                  │
                                                └── retry ◄──────┤ (if all citations dropped, max 1 retry)
                                                                   ▼
    off_topic / injection -> clarify ─────────────────────> END

Design mirrors a miniature grounded-evaluation pipeline: LLM nodes produce
candidate output, a deterministic node verifies citations verbatim before the
answer is marked grounded.  The self-correction loop lets the LLM retry with
a stricter prompt once before falling back to an insufficient-evidence
response.

Multi-turn conversation: the graph accumulates a compact Q&A history per
thread so that follow-up questions ("What about admin accounts?") resolve
correctly via the MemorySaver checkpointer.

Safety: a lightweight guard node screens every input for prompt-injection
patterns before classify runs, refusing obviously adversarial prompts.
"""

import functools
import re
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from opentelemetry import metrics, trace

from . import llm, retriever

_MAX_RETRY_ATTEMPTS = 1

tracer = trace.get_tracer("policy-qa.graph")
meter = metrics.get_meter("policy-qa")
qa_invocations = meter.create_counter(
    "qa.invocations", description="Policy QA graph invocations"
)
valid_citations = meter.create_counter(
    "qa.citations.valid", description="Citations that survived verbatim validation"
)


def _span_attrs(result: dict) -> dict:
    attrs = {}
    if "intent" in result:
        attrs["graph.intent"] = result["intent"]
    if "retrieved" in result:
        attrs["graph.retrieved_count"] = len(result["retrieved"])
    if "citations" in result:
        attrs["graph.citations_count"] = len(result["citations"])
    if "grounded" in result:
        attrs["graph.grounded"] = result["grounded"]
    return attrs


def traced(name: str) -> Callable:
    """Wrap a graph node in an OTel span carrying its state deltas."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapped(state: QAState) -> dict:
            with tracer.start_as_current_span(f"graph.node.{name}") as span:
                result = fn(state)
                for key, value in _span_attrs(result).items():
                    span.set_attribute(key, value)
                return result

        return wrapped

    return deco

CLARIFY_MESSAGE = (
    "This assistant only answers questions about the bundled policy documents "
    "(access control, incident response, and data retention). Please ask a "
    "policy-related question."
)

INSUFFICIENT_EVIDENCE = (
    "The available policy documents do not contain sufficient evidence to "
    "answer this question."
)

INJECTION_REFUSAL = (
    "Your input appears to contain an instruction that conflicts with this "
    "assistant's purpose. This service only answers questions about the "
    "bundled policy documents. Please rephrase your question."
)

_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|your\s+|the\s+)*(?:previous|prior|above)\s+(?:instructions|rules|prompts)",
    r"ignore\s+(?:all\s+)?(?:instructions|rules|prompts)",
    r"(disregard|forget|override)\s+(?:all\s+|your\s+|the\s+)*(?:previous|prior)?\s*(?:instructions|rules|prompts)",
    r"you\s+are\s+now\s+(?:a|an|in)\s+",
    r"system\s*prompt",
    r"\bdo\s+anything\s+now\b",
    r"\bjailbreak\b",
    r"\bdan\s+mode\b",
    r"pretend\s+(?:you\s+are|to\s+be)\s+",
    r"reveal\s+(your|the)\s+(instructions|system|prompt)",
    r"output\s+(your|the)\s+(system|initial|original)\s+(prompt|instructions)",
]


def _append_history(existing: list[dict], new: list[dict]) -> list[dict]:
    """Reducer: append new entries, keep last 10 for token budget."""
    return (existing + new)[-10:]


class QAState(TypedDict, total=False):
    question: str
    intent: str
    retrieved: list[dict]
    answer: str
    citations: list[dict]
    grounded: bool
    error: str | None
    history: Annotated[list[dict], _append_history]
    retry_count: int


def _history_context(state: QAState) -> str:
    """Format recent Q&A history into a short context block."""
    history = state.get("history", [])
    if not history:
        return ""
    lines = []
    for turn in history[-5:]:
        lines.append(f"Q: {turn.get('q', '')}")
        lines.append(f"A: {turn.get('a', '')}")
    return "Recent conversation:\n" + "\n".join(lines) + "\n\n"


def guard(state: QAState) -> dict:
    """Screen input for prompt-injection patterns before classification."""
    text = state.get("question", "").lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text):
            return {"intent": "injection"}
    return {}


def guard_route(state: QAState) -> str:
    """Route injection detections straight to clarify."""
    return "clarify_injection" if state.get("intent") == "injection" else "classify"


def clarify_injection(state: QAState) -> dict:
    return {
        "answer": INJECTION_REFUSAL,
        "citations": [],
        "grounded": False,
        "retrieved": [],
        "history": [{"q": state.get("question", ""), "a": INJECTION_REFUSAL}],
    }


_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["policy_question", "off_topic"]},
    },
    "required": ["intent"],
    "additionalProperties": False,
}


def classify(state: QAState) -> dict:
    hist = _history_context(state)
    system = (
        "You classify user questions. Reply with json only: "
        '{"intent": "policy_question"} if the question is about corporate '
        "policies such as access control, passwords, MFA, incident response, "
        'or data retention; otherwise {"intent": "off_topic"}.'
    )
    user_msg = f"{hist}Current question: {state['question']}"
    result = llm.chat_structured(system, user_msg, _CLASSIFY_SCHEMA, "classify")
    intent = result.get("intent")
    if intent not in ("policy_question", "off_topic"):
        intent = "off_topic"
    return {"intent": intent, "retry_count": 0}


def route(state: QAState) -> str:
    return "retrieve" if state["intent"] == "policy_question" else "clarify"


def retrieve(state: QAState) -> dict:
    return {"retrieved": retriever.retrieve(state["question"])}


_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "relevance": {"type": "integer"},
                },
                "required": ["index", "relevance"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def rerank(state: QAState) -> dict:
    """LLM-based re-ranking: score all chunks in one batched call."""
    retrieved = state.get("retrieved", [])
    if len(retrieved) <= 1:
        return {}
    question = state["question"]
    excerpts = "\n\n".join(
        f"[{i}] ({c['doc']}, section: {c.get('section', 'N/A')})\n{c['text']}"
        for i, c in enumerate(retrieved)
    )
    system = (
        "You are a relevance judge. Given a user question and numbered text "
        "excerpts, rate each excerpt's relevance from 0 to 10. Reply with "
        "json only: {\"scores\": [{\"index\": 0, \"relevance\": <int>}, ...]}."
    )
    user_msg = f"Question: {question}\n\nExcerpts:\n{excerpts}"
    result = llm.chat_structured(system, user_msg, _RERANK_SCHEMA, "rerank")
    scores_list = result.get("scores", [])
    # Build index → relevance map; default to 5 for missing entries
    score_map: dict[int, int] = {}
    for entry in scores_list:
        try:
            idx = int(entry.get("index", -1))
            rel = int(entry.get("relevance", 5))
            score_map[idx] = max(0, min(10, rel))
        except (TypeError, ValueError):
            continue
    scored = []
    for i, chunk in enumerate(retrieved):
        scored.append({**chunk, "relevance": score_map.get(i, 5)})
    # Keep chunks scoring >= 4 (out of 10), sorted descending
    kept = sorted(
        [c for c in scored if c["relevance"] >= 4],
        key=lambda c: c["relevance"],
        reverse=True,
    )
    # Always keep at least 1 chunk
    if not kept and scored:
        kept = [max(scored, key=lambda c: c["relevance"])]
    return {"retrieved": kept}


_GENERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["doc", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


def generate(state: QAState) -> dict:
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "citations": [],
            "grounded": False,
        }
    context = "\n\n".join(
        f"[{c['doc']}] (section: {c.get('section', 'N/A')})\n{c['text']}"
        for c in retrieved
    )
    hist = _history_context(state)
    is_retry = state.get("retry_count", 0) > 0
    retry_warning = (
        "IMPORTANT: Your previous answer had citations that were NOT verbatim "
        "from the excerpts. This time, copy quotes EXACTLY character-for-character "
        "from the excerpt text. Do not paraphrase or modify whitespace.\n\n"
        if is_retry
        else ""
    )
    system = (
        "You answer questions strictly from the provided policy excerpts. "
        "Reply with json only, in this shape: "
        '{"answer": "...", "citations": [{"doc": "<filename>", '
        '"quote": "<verbatim sentence copied from that excerpt>"}]}. '
        "Every claim in the answer must be supported by a verbatim quote "
        "copied exactly from an excerpt. If the excerpts do not answer the "
        'question, reply {"answer": "not_answerable", "citations": []}.'
    )
    user = f"{retry_warning}{hist}Question: {state['question']}\n\nExcerpts:\n{context}"
    result = llm.chat_structured(system, user, _GENERATE_SCHEMA, "generate")
    answer = result.get("answer", "")
    citations = result.get("citations")
    if not answer or answer == "not_answerable" or not isinstance(citations, list):
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "citations": [],
            "grounded": False,
        }
    clean = [
        {"doc": str(c.get("doc", "")), "quote": str(c.get("quote", ""))}
        for c in citations
        if isinstance(c, dict)
    ]
    return {"answer": str(answer), "citations": clean}


def validate(state: QAState) -> dict:
    """Drop citations whose quote is not verbatim in the cited document."""
    valid = [
        c
        for c in state.get("citations", [])
        if c["quote"] and retriever.quote_in_doc(c["doc"], c["quote"])
    ]
    grounded = bool(valid) and state.get("answer") != INSUFFICIENT_EVIDENCE
    if not valid:
        retry = state.get("retry_count", 0)
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "citations": [],
            "grounded": False,
            "retry_count": retry + 1,
        }
    # Append successful turn to history
    q = state.get("question", "")
    a = state.get("answer", "")
    return {
        "citations": valid,
        "grounded": grounded,
        "history": [{"q": q, "a": a}],
    }


def validate_route(state: QAState) -> str:
    """After validate: retry generate if citations were all dropped and retries remain."""
    if state.get("grounded"):
        return END
    if state.get("retry_count", 0) <= _MAX_RETRY_ATTEMPTS:
        return "generate"
    return END


def clarify(state: QAState) -> dict:
    return {
        "answer": CLARIFY_MESSAGE,
        "citations": [],
        "grounded": False,
        "retrieved": [],
        "history": [{"q": state.get("question", ""), "a": CLARIFY_MESSAGE}],
    }


def build_graph() -> Any:
    builder = StateGraph(QAState)
    builder.add_node("guard", traced("guard")(guard))
    builder.add_node("classify", traced("classify")(classify))
    builder.add_node("retrieve", traced("retrieve")(retrieve))
    builder.add_node("rerank", traced("rerank")(rerank))
    builder.add_node("generate", traced("generate")(generate))
    builder.add_node("validate", traced("validate")(validate))
    builder.add_node("clarify", traced("clarify")(clarify))
    builder.add_node("clarify_injection", traced("clarify_injection")(clarify_injection))

    builder.add_edge(START, "guard")
    builder.add_conditional_edges("guard", guard_route, ["classify", "clarify_injection"])
    builder.add_conditional_edges("classify", route, ["retrieve", "clarify"])
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges("validate", validate_route, ["generate", END])
    builder.add_edge("clarify", END)
    builder.add_edge("clarify_injection", END)
    return builder.compile(checkpointer=MemorySaver())
