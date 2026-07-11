"""Export the true holdout set as CSV files for honest model testing.

Recreates the exact train/test split used in training (same seed), so every
row in these files is data the model has NEVER seen. Upload them in the demo
UI's batch-scan tab.

    python scripts/export_holdout.py
        -> samples/holdout_small.csv   (500 unseen txns, 10 unseen frauds)
        -> samples/holdout_full.csv    (the entire unseen test set)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from fraud_detection.config import load_config  # noqa: E402
from fraud_detection.features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402


def main() -> None:
    cfg = load_config()
    raw = Path(cfg.data.raw_path)
    if not raw.exists():
        raise SystemExit("Dataset not cached - run `make train` first")

    df = pd.read_parquet(raw)
    x, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    # Identical split parameters to train.py -> identical held-out rows.
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.data.random_state
    )
    holdout = x_test.copy()
    holdout["actual_fraud"] = y_test.values

    out = Path("samples")
    out.mkdir(exist_ok=True)

    frauds = holdout[holdout["actual_fraud"] == 1]
    small = pd.concat(
        [
            holdout[holdout["actual_fraud"] == 0].sample(490, random_state=1),
            frauds.sample(10, random_state=1),
        ]
    ).sample(frac=1.0, random_state=1)
    small.to_csv(out / "holdout_small.csv", index=False)
    holdout.to_csv(out / "holdout_full.csv", index=False)

    print(f"samples/holdout_small.csv  {len(small):>7,} rows, {int(small['actual_fraud'].sum())} frauds")
    print(f"samples/holdout_full.csv   {len(holdout):>7,} rows, {int(holdout['actual_fraud'].sum())} frauds")
    print("\nEvery row here was excluded from training - this is the honest test.")


if __name__ == "__main__":
    main()
