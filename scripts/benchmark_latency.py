"""Measure prediction latency: the model alone, and a full HTTP request.

Start the API without prediction logging, then run this against it:

    uvicorn fraud_detection.api.main:app --port 8000
    python scripts/benchmark_latency.py --n 2000

Requests are sent one at a time, so this measures latency, not throughput
under load. Results depend on the machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import httpx
import numpy as np
import pandas as pd
import xgboost as xgb

REFERENCE_PATH = "models/reference_sample.parquet"
WARMUP_REQUESTS = 100


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    return {f"p{p}": round(float(np.percentile(samples_ms, p)), 2) for p in (50, 95, 99)}


def benchmark_model(rows: list[dict], features: list[str]) -> list[float]:
    """Time the same steps as /predict: build the row, then score it."""
    booster = xgb.Booster()
    booster.load_model("models/model.json")
    samples = []
    for row in rows:
        start = time.perf_counter()
        matrix = xgb.DMatrix(
            np.array([[row[f] for f in features]], dtype=np.float32), feature_names=features
        )
        booster.predict(matrix)
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def benchmark_http(rows: list[dict], url: str) -> list[float]:
    samples = []
    with httpx.Client(base_url=url, timeout=10) as client:
        for row in rows[:WARMUP_REQUESTS]:
            client.post("/predict", json=row).raise_for_status()
        for row in rows:
            start = time.perf_counter()
            client.post("/predict", json=row).raise_for_status()
            samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    features = json.load(open("models/metadata.json"))["features"]
    batch = pd.read_parquet(REFERENCE_PATH).sample(args.n, replace=True, random_state=args.seed)
    rows = [{f: float(row[f]) for f in features} for row in batch.to_dict(orient="records")]

    print(json.dumps({
        "machine": f"{platform.system()} {platform.machine()}",
        "requests": args.n,
        "model_only_ms": percentiles(benchmark_model(rows, features)),
        "http_round_trip_ms": percentiles(benchmark_http(rows, args.url)),
    }, indent=2))


if __name__ == "__main__":
    main()
