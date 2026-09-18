"""Emit eval results to Application Insights as spans.

Each eval check becomes a span named ``eval::<nodeid>`` carrying
``eval.passed``, optional ``eval.score`` and ``eval.reason`` attributes, so
results are queryable in App Insights Logs and chartable in Workbooks.
No-ops when APPLICATIONINSIGHTS_CONNECTION_STRING is unset.
"""

import os

_provider = None
_tracer = None


def _get_tracer():
    global _provider, _tracer
    if _tracer is not None:
        return _tracer
    cs = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    if not cs:
        return None
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    _provider = TracerProvider()
    _provider.add_span_processor(
        SimpleSpanProcessor(AzureMonitorTraceExporter(connection_string=cs))
    )
    trace.set_tracer_provider(_provider)
    _tracer = _provider.get_tracer("policy-qa.evals")
    return _tracer


def record_eval(name: str, passed: bool, score: float | None = None, reason: str = "") -> None:
    tracer = _get_tracer()
    if tracer is None:
        return
    span = tracer.start_span(f"eval::{name}")
    span.set_attribute("eval.passed", passed)
    if score is not None:
        span.set_attribute("eval.score", float(score))
    if reason:
        span.set_attribute("eval.reason", reason[:500])
    span.end()


def flush() -> None:
    if _provider is not None:
        _provider.force_flush()
