"""Training pipeline: load -> split -> train XGBoost -> evaluate -> log to MLflow.

Usage:
    python -m fraud_detection.train                 # real dataset (downloads once)
    python -m fraud_detection.train --synthetic     # fast run on synthetic data (CI)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fraud_detection.config import Config, load_config
from fraud_detection.data import FEATURE_COLUMNS, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate(model: XGBClassifier, x_test, y_test, threshold: float) -> dict[str, float]:
    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "precision_at_threshold": float(precision_score(y_test, preds, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_test, preds, zero_division=0)),
        "fraud_rate_test": float(np.mean(y_test)),
    }


def train(cfg: Config, synthetic: bool = False, out_dir: str | None = None) -> dict[str, float]:
    """Train and export a model. `out_dir` overrides the export location
    (used by retraining to stage a challenger without touching the champion)."""
    x, y = load_dataset(cfg.data.openml_id, cfg.data.raw_path, synthetic=synthetic)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.data.random_state
    )

    # Handle class imbalance via positive-class weighting.
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {**cfg.model.params, "scale_pos_weight": scale_pos_weight}

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        model = XGBClassifier(**params)
        model.fit(x_train, y_train)

        metrics = evaluate(model, x_test, y_test, cfg.serving.decision_threshold)

        mlflow.log_params({**params, "synthetic_data": synthetic, "n_train": len(x_train)})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model.get_booster(),
            name="model",
            registered_model_name=None if synthetic else cfg.mlflow.registered_model_name,
        )

        # Export a serving copy: plain XGBoost JSON + metadata (no MLflow needed at inference).
        model_dir = Path(out_dir or cfg.serving.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.get_booster().save_model(model_dir / "model.json")
        # Reference sample for drift detection (PSI baseline).
        x_train.sample(min(10_000, len(x_train)), random_state=42).to_parquet(
            model_dir / "reference_sample.parquet"
        )
        (model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "features": FEATURE_COLUMNS,
                    "threshold": cfg.serving.decision_threshold,
                    "metrics": metrics,
                    "synthetic_data": synthetic,
                },
                indent=2,
            )
        )

        logger.info("Run %s | %s", run.info.run_id, json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (fast, CI)")
    args = parser.parse_args()
    train(load_config(args.config), synthetic=args.synthetic)


if __name__ == "__main__":
    main()
