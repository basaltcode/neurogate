from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

registry = CollectorRegistry()

requests_total = Counter(
    "neurogate_requests_total",
    "Provider call outcomes",
    labelnames=("provider", "outcome"),
    registry=registry,
)

request_duration_seconds = Histogram(
    "neurogate_request_duration_seconds",
    "Provider call latency in seconds",
    labelnames=("provider", "outcome"),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0),
    registry=registry,
)


