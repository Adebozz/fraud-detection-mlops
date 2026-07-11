"""Send simulated transactions at the API - normal or drifted.

Normal mode samples real rows from the training reference sample, so traffic
matches the training distribution and PSI stays "ok". Drift mode shifts a
subset of features (V1, V3, V14, Amount) so only those get flagged.

    python scripts/simulate_traffic.py --n 300
    python scripts/simulate_traffic.py --n 300 --drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

DRIFTED_FEATURES = ["V1", "V3", "V14"]
REFERENCE_PATH = "models/reference_sample.parquet"


def build_batch(n: int, drift: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if Path(REFERENCE_PATH).exists():
        ref = pd.read_parquet(REFERENCE_PATH)
        batch = ref.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    else:  # fallback if no trained model yet (synthetic-schema normals)
        print(f"WARNING: {REFERENCE_PATH} not found, using synthetic distributions")
        batch = pd.DataFrame({f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)})
        batch["Amount"] = np.round(rng.lognormal(3.0, 1.2, n), 2)

    if drift:
        for col in DRIFTED_FEATURES:
            batch[col] = batch[col] + 3.0 * batch[col].std()
        batch["Amount"] = np.round(batch["Amount"] * 4.0, 2)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    batch = build_batch(args.n, args.drift, args.seed)
    frauds = errors = 0
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for i, row in enumerate(batch.to_dict(orient="records")):
            r = client.post("/predict", json={k: float(v) for k, v in row.items()})
            if r.status_code != 200:
                errors += 1
                continue
            frauds += r.json()["is_fraud"]
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{args.n} sent...")

    mode = f"DRIFTED ({', '.join(DRIFTED_FEATURES)}, Amount)" if args.drift else "normal"
    print(f"Done ({mode}): {args.n} requests, {frauds} flagged fraud, {errors} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
