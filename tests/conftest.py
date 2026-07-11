"""Shared fixtures: a tiny trained model exported the same way train.py does it."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    from fraud_detection.data import FEATURE_COLUMNS, TARGET_COLUMN, make_synthetic

    df = make_synthetic(n_rows=2_000, fraud_rate=0.05, seed=0)
    x, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

    model = XGBClassifier(n_estimators=20, max_depth=3, eval_metric="aucpr")
    model.fit(x_train, y_train)

    out = tmp_path_factory.mktemp("model")
    model.get_booster().save_model(out / "model.json")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "features": FEATURE_COLUMNS,
                "threshold": 0.5,
                "metrics": {},
                "synthetic_data": True,
            }
        )
    )
    return out
