"""Unit tests for the retriever module (keyword retrieval + validation)."""

import pytest

from app.retriever import (
    _keyword_retrieve,
    _tokenize,
    load_chunks,
    load_documents,
    normalize_ws,
    quote_in_doc,
)


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello, World! 123")
        assert tokens == {"hello", "world", "123"}

    def test_empty(self):
        assert _tokenize("") == set()
        assert _tokenize("!!! ???") == set()


class TestNormalizeWs:
    def test_collapses_whitespace(self):
        assert normalize_ws("foo  bar\n\tbaz") == "foo bar baz"

    def test_strips(self):
        assert normalize_ws("  hello  ") == "hello"

    def test_empty(self):
        assert normalize_ws("") == ""


class TestLoadDocuments:
    def test_returns_dict(self):
        docs = load_documents()
        assert isinstance(docs, dict)
        assert len(docs) == 3

    def test_expected_docs_present(self):
        docs = load_documents()
        assert "access_control_policy.md" in docs
        assert "incident_response_policy.md" in docs
        assert "data_retention_policy.md" in docs


class TestLoadChunks:
    def test_returns_list(self):
        chunks = load_chunks()
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_has_expected_keys(self):
        chunk = load_chunks()[0]
        assert "doc" in chunk
        assert "text" in chunk
        assert "section" in chunk


class TestQuoteInDoc:
    def test_verbatim_match(self):
        assert quote_in_doc(
            "access_control_policy.md",
            "Passwords must be rotated every 90 days.",
        )

    def test_whitespace_normalization(self):
        assert quote_in_doc(
            "access_control_policy.md",
            "Passwords  must  be  rotated  every  90  days.",
        )

    def test_nonexistent_doc(self):
        assert not quote_in_doc("nonexistent.md", "anything")

    def test_wrong_quote(self):
        assert not quote_in_doc(
            "access_control_policy.md",
            "This quote does not exist in the document at all.",
        )


class TestKeywordRetrieve:
    def test_returns_results_for_policy_question(self):
        results = _keyword_retrieve("password rotation policy", top_k=3)
        assert len(results) > 0
        assert all("score" in r for r in results)
        assert results[0]["score"] >= results[-1]["score"]

    def test_empty_query(self):
        assert _keyword_retrieve("", top_k=3) == []

    def test_respects_top_k(self):
        results = _keyword_retrieve("policy", top_k=1)
        assert len(results) <= 1

    def test_results_have_correct_keys(self):
        results = _keyword_retrieve("access control", top_k=2)
        for r in results:
            assert "doc" in r
            assert "text" in r
            assert "section" in r
            assert "score" in r
