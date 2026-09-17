"""Deterministic contract checks for the deployed policy Q&A service.

These tests own the hard guarantees — schema, routing, citation veracity, and
refusal behavior — and must pass 100%. No LLM judge is involved.
"""

import re
from pathlib import Path

import pytest

from conftest import GOLDENS, responses  # noqa: F401  (fixture import)

DOCS_DIR = Path(__file__).parent.parent / "app" / "data"


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_response_schema(case, responses):
    body = responses[case["id"]]["response"]
    assert set(body) >= {
        "thread_id", "intent", "answer", "citations", "grounded",
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
        doc_text = _normalized(doc_path.read_text())
        assert _normalized(citation["quote"]) in doc_text, (
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
