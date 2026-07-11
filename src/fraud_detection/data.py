"""Data acquisition, validation, and synthetic generation.

The real dataset is the ULB credit-card fraud dataset (OpenML id 1597):
284,807 transactions, 29 features (V1-V28 PCA components, Amount), binary
target `Class` (0.17% positive). The OpenML variant omits the original Time
column. The synthetic generator mirrors the schema and class imbalance so
tests and CI never need the real download.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [*[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"


def download_openml(openml_id: int, out_path: str | Path) -> pd.DataFrame:
    """Download the dataset from OpenML and cache it as parquet."""
    out_path = Path(out_path)
    if out_path.exists():
        logger.info("Using cached dataset at %s", out_path)
        return pd.read_parquet(out_path)

    from sklearn.datasets import fetch_openml  # local import: heavy

    logger.info("Downloading OpenML dataset %s (~150 MB, one-off)...", openml_id)
    bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    df = bunch.frame
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info("Cached %d rows to %s", len(df), out_path)
    return df


def make_synthetic(n_rows: int = 20_000, fraud_rate: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Generate schema-compatible synthetic transactions.

    Fraud rows are drawn from shifted distributions so a model can learn a
    real (if easy) signal - enough to exercise the full pipeline in CI.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_rate), 10)
    n_legit = n_rows - n_fraud

    def block(n: int, shift: float) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for i in range(1, 29):
            cols[f"V{i}"] = rng.normal(loc=shift * (i % 3), scale=1.0, size=n)
        cols["Amount"] = np.round(rng.lognormal(mean=3.0 + shift, sigma=1.2, size=n), 2)
        return cols

    legit = pd.DataFrame(block(n_legit, shift=0.0))
    legit[TARGET_COLUMN] = 0
    fraud = pd.DataFrame(block(n_fraud, shift=1.5))
    fraud[TARGET_COLUMN] = 1

    df = pd.concat([legit, fraud], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Fail fast on schema or quality problems before training."""
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("NaNs found in feature columns")
    if not set(df[TARGET_COLUMN].unique()) <= {0, 1}:
        raise ValueError("Target must be binary 0/1")
    if df[TARGET_COLUMN].sum() == 0:
        raise ValueError("No positive (fraud) examples in dataset")
    if (df["Amount"] < 0).any():
        raise ValueError("Negative transaction amounts found")
    logger.info(
        "Validation passed: %d rows, fraud rate %.4f%%",
        len(df),
        100 * df[TARGET_COLUMN].mean(),
    )


def load_dataset(
    openml_id: int, raw_path: str | Path, synthetic: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y), downloading or generating as needed."""
    df = make_synthetic() if synthetic else download_openml(openml_id, raw_path)
    validate(df)
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]
