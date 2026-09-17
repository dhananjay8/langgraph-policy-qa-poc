"""Azure OpenAI client wrapper for the policy Q&A graph."""

import json
import os
from typing import Any

from openai import AzureOpenAI

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


def chat_json(system: str, user: str) -> dict[str, Any]:
    """Call the chat deployment and parse a JSON object response.

    Returns an empty dict on any failure so callers can apply a deterministic
    fallback instead of crashing the graph.
    """
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    try:
        resp = get_client().chat.completions.create(
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
        return {}
