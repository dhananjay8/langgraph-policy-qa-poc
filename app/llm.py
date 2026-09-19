"""Azure OpenAI client wrapper for the policy Q&A graph.

Provides ``chat_json`` (free-form JSON mode) and ``chat_structured``
(JSON-schema-constrained) helpers, both with retry-with-backoff for
transient Azure errors.
"""

import json
import logging
import os
from typing import Any

from openai import AzureOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("policy-qa.llm")

_client: AzureOpenAI | None = None


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    return _client


# Retry on transient OpenAI / network errors (3 attempts, 1-4s backoff)
_RETRYABLE = (Exception,)  # broad; narrowed below if openai exposes types
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError
    _RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)
except ImportError:
    pass


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _completions_create(**kwargs):
    """Thin wrapper so tenacity retries transient failures."""
    return get_client().chat.completions.create(**kwargs)


def chat_json(system: str, user: str) -> dict[str, Any]:
    """Call the chat deployment and parse a JSON object response.

    Returns an empty dict on any failure so callers can apply a deterministic
    fallback instead of crashing the graph.
    """
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    try:
        resp = _completions_create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=4000,
        )
        content = resp.choices[0].message.content or ""
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception:
        log.exception("chat_json call failed after retries")
        return {}


def chat_structured(
    system: str,
    user: str,
    json_schema: dict[str, Any],
    schema_name: str = "response",
) -> dict[str, Any]:
    """Call the chat deployment with a strict JSON schema constraint.

    Uses ``response_format: {type: json_schema, ...}`` so the model is
    constrained to output only valid instances of *json_schema*.  Falls back
    to ``chat_json`` if the deployment does not support structured output.
    """
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    try:
        resp = _completions_create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
            max_completion_tokens=4000,
        )
        content = resp.choices[0].message.content or ""
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception:
        log.warning("chat_structured failed, falling back to chat_json")
        return chat_json(system, user)
