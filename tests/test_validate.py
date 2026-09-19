"""Unit tests for the validate node and validate_route logic."""

import pytest

from app.graph import (
    INSUFFICIENT_EVIDENCE,
    _MAX_RETRY_ATTEMPTS,
    validate,
    validate_route,
)
from langgraph.graph import END


class TestValidate:
    def test_valid_citation_kept(self):
        state = {
            "question": "password length?",
            "answer": "14 characters",
            "citations": [
                {
                    "doc": "access_control_policy.md",
                    "quote": "All user passwords must be at least 14 characters long",
                }
            ],
        }
        result = validate(state)
        assert result["grounded"] is True
        assert len(result["citations"]) == 1

    def test_invalid_citation_dropped(self):
        state = {
            "question": "password length?",
            "answer": "14 characters",
            "citations": [
                {
                    "doc": "access_control_policy.md",
                    "quote": "This quote does not exist in the document",
                }
            ],
            "retry_count": 0,
        }
        result = validate(state)
        assert result["grounded"] is False
        assert result["citations"] == []
        assert result["answer"] == INSUFFICIENT_EVIDENCE
        assert result["retry_count"] == 1

    def test_mixed_citations_partial_survival(self):
        state = {
            "question": "password rotation?",
            "answer": "Rotated every 90 days.",
            "citations": [
                {
                    "doc": "access_control_policy.md",
                    "quote": "Passwords must be rotated every 90 days.",
                },
                {
                    "doc": "access_control_policy.md",
                    "quote": "This fake quote is not in the document.",
                },
            ],
        }
        result = validate(state)
        assert result["grounded"] is True
        assert len(result["citations"]) == 1
        assert result["citations"][0]["quote"] == "Passwords must be rotated every 90 days."

    def test_empty_quote_dropped(self):
        state = {
            "question": "test",
            "answer": "test",
            "citations": [{"doc": "access_control_policy.md", "quote": ""}],
            "retry_count": 0,
        }
        result = validate(state)
        assert result["grounded"] is False

    def test_history_appended_on_success(self):
        state = {
            "question": "What is MFA?",
            "answer": "Hardware-based MFA tokens are required.",
            "citations": [
                {
                    "doc": "access_control_policy.md",
                    "quote": "Administrative accounts must use hardware-based MFA tokens.",
                }
            ],
        }
        result = validate(state)
        assert "history" in result
        assert result["history"][0]["q"] == "What is MFA?"


class TestValidateRoute:
    def test_grounded_goes_to_end(self):
        assert validate_route({"grounded": True, "retry_count": 0}) == END

    def test_not_grounded_retries_when_allowed(self):
        assert validate_route({"grounded": False, "retry_count": 0}) == "generate"
        assert validate_route({"grounded": False, "retry_count": 1}) == "generate"

    def test_not_grounded_ends_when_retries_exhausted(self):
        assert validate_route({"grounded": False, "retry_count": _MAX_RETRY_ATTEMPTS + 1}) == END
