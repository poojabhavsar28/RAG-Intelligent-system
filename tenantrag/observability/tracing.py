"""Optional OpenTelemetry tracing. Without ``opentelemetry`` installed every call is a no-op,
so the engine never depends on it. Span attributes carry ids, counts and timings only,
never query text or answer content."""
from __future__ import annotations

from contextlib import contextmanager

try:  # pragma: no cover - exercised only when otel is installed
    from opentelemetry import trace

    _tracer = trace.get_tracer("tenantrag")
except Exception:  # noqa: BLE001
    _tracer = None


@contextmanager
def span(name: str, **attrs):
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:  # pragma: no cover
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield s
