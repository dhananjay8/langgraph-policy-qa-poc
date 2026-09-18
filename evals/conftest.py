import json
import os
import uuid
from pathlib import Path

import httpx
import pytest

GOLDENS = json.loads((Path(__file__).parent / "golden_dataset.json").read_text())

BASE_URL = os.environ.get("POC_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("POC_API_KEY", "")


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        headers={"x-api-key": API_KEY},
        timeout=90.0,
    )


@pytest.fixture(scope="session")
def responses(client) -> dict:
    """Call the deployed service once per golden case and cache responses."""
    out = {}
    for case in GOLDENS:
        thread = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
        r = client.post(f"/v1/qa/{thread}", json={"question": case["question"]})
        r.raise_for_status()
        out[case["id"]] = {"case": case, "response": r.json()}
    return out


def golden(case_id: str) -> dict:
    return next(c for c in GOLDENS if c["id"] == case_id)


def pytest_runtest_logreport(report):
    """Stream every eval outcome to App Insights (no-op without a conn string)."""
    if report.when != "call":
        return
    import telemetry

    score = dict(report.user_properties).get("eval_score")
    reason = report.longreprtext[-400:] if report.failed else ""
    telemetry.record_eval(report.nodeid, report.passed, score=score, reason=reason)


def pytest_sessionfinish(session):
    import telemetry

    telemetry.flush()
