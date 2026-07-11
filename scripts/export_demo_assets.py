"""Export small demo batch files for cloud deployment.

The hosted demo (Streamlit Community Cloud) has no access to the 150 MB
dataset, so we commit three small sample batches to the repo. Run once
locally, then commit the samples/ CSVs.

    python scripts/export_demo_assets.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/creditcard.parquet")
OUT = Path("samples")

BATCHES = {  # name: (seed, n_legit, n_fraud, high_value)
    "demo_batch_a.csv": (1, 196, 4, False),
    "demo_batch_b.csv": (2, 488, 12, False),
    "demo_batch_c.csv": (3, 97, 3, True),
}


def main() -> None:
    if not RAW.exists():
        raise SystemExit("Dataset not cached - run `make train` first")
    df = pd.read_parquet(RAW)
    legit_all, fraud = df[df["Class"] == 0], df[df["Class"] == 1]
    OUT.mkdir(exist_ok=True)

    for name, (seed, n_legit, n_fraud, high_value) in BATCHES.items():
        legit = legit_all
        if high_value:
            legit = legit[legit["Amount"] > legit["Amount"].quantile(0.9)]
        batch = pd.concat(
            [legit.sample(n_legit, random_state=seed), fraud.sample(n_fraud, random_state=seed)]
        ).sample(frac=1.0, random_state=seed)
        batch = batch.rename(columns={"Class": "actual_fraud"}).reset_index(drop=True)
        batch.insert(0, "transaction_id", [f"TXN-{seed}{i:05d}" for i in range(len(batch))])
        batch.to_csv(OUT / name, index=False)
        print(f"{OUT / name}: {len(batch)} rows, {int(batch['actual_fraud'].sum())} frauds")


if __name__ == "__main__":
    main()
