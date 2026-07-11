"""Feature-drift detection via Population Stability Index (PSI).

PSI is the standard drift metric in credit/fraud risk:
    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Conventional thresholds:
    < 0.10        no significant change
    0.10 - 0.25   moderate shift, investigate
    > 0.25        significant drift, retrain

Usage:
    python -m fraud_detection.drift \
        --reference models/reference_sample.parquet \
        --current "logs/predictions-*.jsonl"

Exit code 1 when any feature exceeds the drift threshold - usable directly
as a CI/cron gate to trigger retraining.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.features import FEATURE_COLUMNS

WARN_THRESHOLD = 0.10
DRIFT_THRESHOLD = 0.25


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI between two samples, binned on the expected (reference) quantiles."""
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("Cannot compute PSI on empty arrays")

    # Quantile bins from the reference distribution; unique() guards constants.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: PSI is 0 unless values moved
        return 0.0 if np.allclose(np.median(expected), np.median(actual)) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected) + eps
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual) + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Per-feature PSI table, sorted worst-first."""
    rows = [
        {"feature": col, "psi": psi(reference[col].to_numpy(), current[col].to_numpy())}
        for col in FEATURE_COLUMNS
        if col in reference.columns and col in current.columns
    ]
    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    report["status"] = pd.cut(
        report["psi"],
        bins=[-np.inf, WARN_THRESHOLD, DRIFT_THRESHOLD, np.inf],
        labels=["ok", "warn", "drift"],
    )
    return report


def load_prediction_logs(pattern: str) -> pd.DataFrame:
    """Flatten JSONL prediction logs (written by the API) into a feature frame."""
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            records.extend(json.loads(line)["features"] for line in f if line.strip())
    if not records:
        raise FileNotFoundError(f"No prediction records found for pattern: {pattern}")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="models/reference_sample.parquet")
    parser.add_argument("--current", default="logs/predictions-*.jsonl")
    parser.add_argument("--out", default="models/drift_report.json")
    args = parser.parse_args()

    reference = pd.read_parquet(args.reference)
    current = load_prediction_logs(args.current)
    report = drift_report(reference, current)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report.to_json(orient="records", indent=2))

    print(report.head(10).to_string(index=False))
    n_drifted = int((report["status"] == "drift").sum())
    print(f"\n{n_drifted} feature(s) above drift threshold ({DRIFT_THRESHOLD}). "
          f"Current sample: {len(current)} rows.")
    sys.exit(1 if n_drifted > 0 else 0)


if __name__ == "__main__":
    main()
