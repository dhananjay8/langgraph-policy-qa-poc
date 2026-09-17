"""LangGraph policy Q&A graph.

Flow:
    classify -> (policy_question | off_topic)
    policy_question -> retrieve -> generate -> validate -> END
    off_topic       -> clarify -> END

Design mirrors a miniature grounded-evaluation pipeline: LLM nodes produce
candidate output, a deterministic node verifies citations verbatim before the
answer is marked grounded.
"""

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import llm, retriever

CLARIFY_MESSAGE = (
    "This assistant only answers questions about the bundled policy documents "
    "(access control, incident response, and data retention). Please ask a "
    "policy-related question."
)

INSUFFICIENT_EVIDENCE = (
    "The available policy documents do not contain sufficient evidence to "
    "answer this question."
)


class QAState(TypedDict, total=False):
    question: str
    intent: str
    retrieved: list[dict]
    answer: str
    citations: list[dict]
    grounded: bool
    error: str | None


def classify(state: QAState) -> dict:
    system = (
        "You classify user questions. Reply with json only: "
        '{"intent": "policy_question"} if the question is about corporate '
        "policies such as access control, passwords, MFA, incident response, "
        'or data retention; otherwise {"intent": "off_topic"}.'
    )
    result = llm.chat_json(system, state["question"])
    intent = result.get("intent")
    if intent not in ("policy_question", "off_topic"):
        intent = "off_topic"
    return {"intent": intent}


def route(state: QAState) -> str:
    return "retrieve" if state["intent"] == "policy_question" else "clarify"


def retrieve(state: QAState) -> dict:
    return {"retrieved": retriever.retrieve(state["question"])}


def generate(state: QAState) -> dict:
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "citations": [],
            "grounded": False,
        }
    context = "\n\n".join(
        f"[{c['doc']}]\n{c['text']}" for c in retrieved
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
    user = f"Question: {state['question']}\n\nExcerpts:\n{context}"
    result = llm.chat_json(system, user)
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
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "citations": [],
            "grounded": False,
        }
    return {"citations": valid, "grounded": grounded}


def clarify(state: QAState) -> dict:
    return {
        "answer": CLARIFY_MESSAGE,
        "citations": [],
        "grounded": False,
        "retrieved": [],
    }


def build_graph() -> Any:
    builder = StateGraph(QAState)
    builder.add_node("classify", classify)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("validate", validate)
    builder.add_node("clarify", clarify)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route, ["retrieve", "clarify"])
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "validate")
    builder.add_edge("validate", END)
    builder.add_edge("clarify", END)
    return builder.compile(checkpointer=MemorySaver())
