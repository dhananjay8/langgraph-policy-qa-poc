"""Deterministic contract checks for the deployed policy Q&A service.

These tests own the hard guarantees — schema, routing, citation veracity, and
refusal behavior — and must pass 100%. No LLM judge is involved.
"""

import sys
from pathlib import Path

import pytest

# Make app importable so we can reuse normalize_ws
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.retriever import normalize_ws  # noqa: E402

from conftest import GOLDENS, responses  # noqa: F401, E402  (fixture import)

DOCS_DIR = Path(__file__).parent.parent / "app" / "data"


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_response_schema(case, responses):
    body = responses[case["id"]]["response"]
    assert set(body) >= {
        "request_id", "thread_id", "intent", "answer", "citations", "grounded",
        "retrieved_docs", "latency_ms",
    }
    assert isinstance(body["citations"], list)
    assert isinstance(body["grounded"], bool)


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_intent_routing(case, responses):
    assert responses[case["id"]]["response"]["intent"] == case["expected_intent"]


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_expected_document_retrieved(case, responses):
    if not case["expected_docs"]:
        return
    retrieved = set(responses[case["id"]]["response"]["retrieved_docs"])
    assert retrieved & set(case["expected_docs"]), (
        f"expected one of {case['expected_docs']} in {retrieved}"
    )


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_citations_are_verbatim(case, responses):
    """Every returned citation quote must be a verbatim substring of its doc."""
    body = responses[case["id"]]["response"]
    for citation in body["citations"]:
        doc_path = DOCS_DIR / citation["doc"]
        assert doc_path.exists(), f"citation references unknown doc {citation['doc']}"
        doc_text = normalize_ws(doc_path.read_text())
        assert normalize_ws(citation["quote"]) in doc_text, (
            f"non-verbatim quote survived validation: {citation['quote'][:80]}"
        )


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_grounded_flag_consistency(case, responses):
    body = responses[case["id"]]["response"]
    assert body["grounded"] == bool(body["citations"])
    if case["must_be_grounded"]:
        assert body["grounded"], f"{case['id']} should have produced citations"


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_answer_keywords(case, responses):
    answer = (responses[case["id"]]["response"]["answer"] or "").lower()
    for kw in case["answer_keywords"]:
        assert kw.lower() in answer, f"missing expected content: {kw}"


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_off_topic_refusal(case, responses):
    if case["expected_intent"] != "off_topic":
        return
    body = responses[case["id"]]["response"]
    assert body["citations"] == []
    assert "policy" in (body["answer"] or "").lower()


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_injection_refusal(case, responses):
    if case["expected_intent"] != "injection":
        return
    body = responses[case["id"]]["response"]
    assert body["citations"] == []
    assert not body["grounded"]
    assert "policy" in (body["answer"] or "").lower()
