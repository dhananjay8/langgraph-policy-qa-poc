"""Unit tests for the FastAPI endpoints (no LLM calls — uses TestClient)."""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set required env vars before importing app
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "fake-key")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
os.environ.setdefault("POC_API_KEY", "test-key")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
HEADERS = {"x-api-key": "test-key"}


class TestRoot:
    def test_returns_service_info(self):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "langgraph-policy-qa-poc"
        assert "version" in body
        assert "endpoints" in body


class TestHealthz:
    def test_unauthenticated_returns_minimal(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "uptime_seconds" in body
        # Should NOT expose internal diagnostics
        assert "llm" not in body

    def test_authenticated_returns_full(self):
        r = client.get("/healthz", headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "documents" in body


class TestAuth:
    def test_missing_api_key_returns_401(self):
        r = client.post("/v1/qa/thread-1", json={"question": "test?"})
        assert r.status_code == 401

    def test_wrong_api_key_returns_401(self):
        r = client.post(
            "/v1/qa/thread-1",
            json={"question": "test?"},
            headers={"x-api-key": "wrong-key"},
        )
        assert r.status_code == 401


class TestInputValidation:
    def test_empty_question_rejected(self):
        r = client.post(
            "/v1/qa/thread-1",
            json={"question": ""},
            headers=HEADERS,
        )
        assert r.status_code == 422

    def test_whitespace_only_question_rejected(self):
        r = client.post(
            "/v1/qa/thread-1",
            json={"question": "   \n\t  "},
            headers=HEADERS,
        )
        assert r.status_code == 422

    def test_missing_question_rejected(self):
        r = client.post(
            "/v1/qa/thread-1",
            json={},
            headers=HEADERS,
        )
        assert r.status_code == 422


class TestRateLimiting:
    def test_rate_limit_returns_429(self):
        """Temporarily set rate limit to 2 and verify 429 on 3rd request."""
        import app.main as main_mod
        original_limit = main_mod._RATE_LIMIT
        main_mod._RATE_LIMIT = 2
        # Clear existing entries for our key
        main_mod._request_log.clear()
        try:
            # Use a unique thread to avoid graph side effects — we just need
            # the rate limiter to fire before the graph runs
            for i in range(2):
                r = client.post(
                    f"/v1/qa/rate-test-{i}",
                    json={"question": "What is the password policy?"},
                    headers=HEADERS,
                )
                # First two may succeed or fail for other reasons,
                # but should not be 429
                assert r.status_code != 429

            # Third request should be rate-limited
            r = client.post(
                "/v1/qa/rate-test-3",
                json={"question": "What is the password policy?"},
                headers=HEADERS,
            )
            assert r.status_code == 429
        finally:
            main_mod._RATE_LIMIT = original_limit
            main_mod._request_log.clear()
