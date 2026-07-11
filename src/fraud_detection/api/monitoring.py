"""Prometheus metrics and prediction logging for the serving API.

Metrics exposed at /metrics:
- fraud_predictions_total{decision="fraud|legit"}   - prediction volume by outcome
- fraud_probability                                  - histogram of scores (drift signal)
- fraud_request_latency_seconds                      - inference latency histogram

Prediction logging: when $PRED_LOG_DIR is set, every request/response pair is
appended as JSONL - the input to offline drift detection (fraud_detection.drift).
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import Counter, Histogram

PREDICTIONS = Counter(
    "fraud_predictions_total", "Number of predictions served", ["decision"]
)
PROBABILITY = Histogram(
    "fraud_probability",
    "Distribution of predicted fraud probabilities",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
)
LATENCY = Histogram(
    "fraud_request_latency_seconds",
    "Model inference latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
SHADOW_PROBABILITY = Histogram(
    "fraud_shadow_probability",
    "Shadow (challenger) model score distribution",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
)
SHADOW_DISAGREEMENTS = Counter(
    "fraud_shadow_disagreements_total",
    "Requests where champion and shadow decisions differ",
)

_write_lock = threading.Lock()


def observe(probability: float, is_fraud: bool, latency_s: float) -> None:
    PREDICTIONS.labels(decision="fraud" if is_fraud else "legit").inc()
    PROBABILITY.observe(probability)
    LATENCY.observe(latency_s)


def observe_shadow(shadow_probability: float, champion_fraud: bool, shadow_fraud: bool) -> None:
    SHADOW_PROBABILITY.observe(shadow_probability)
    if champion_fraud != shadow_fraud:
        SHADOW_DISAGREEMENTS.inc()


def log_prediction(
    features: dict, probability: float, is_fraud: bool, shadow_probability: float | None = None
) -> None:
    """Append the prediction to a daily JSONL file if PRED_LOG_DIR is set."""
    log_dir = os.environ.get("PRED_LOG_DIR")
    if not log_dir:
        return
    path = Path(log_dir) / f"predictions-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "features": features,
        "fraud_probability": probability,
        "is_fraud": is_fraud,
    }
    if shadow_probability is not None:
        record["shadow_probability"] = shadow_probability
    with _write_lock, path.open("a") as f:
        f.write(json.dumps(record) + "\n")
