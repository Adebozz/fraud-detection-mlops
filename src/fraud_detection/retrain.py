"""Champion/challenger retraining.

    python -m fraud_detection.retrain              # train challenger -> models/shadow, compare
    python -m fraud_detection.retrain --promote    # promote staged challenger if it's better
    python -m fraud_detection.retrain --synthetic  # fast CI variant

Flow: a drift alert (fraud_detection.drift exits 1) triggers retraining. The
challenger is staged to models/shadow - never straight to production. Deploy
it in shadow mode (SHADOW_MODEL_DIR) to score live traffic silently, then
promote once you're satisfied. Promotion requires the challenger to beat the
champion on PR-AUC.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from fraud_detection.config import Config, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_METRIC = "pr_auc"
ARTIFACTS = ["model.json", "metadata.json", "reference_sample.parquet"]


def read_metrics(model_dir: str | Path) -> dict[str, float] | None:
    meta_path = Path(model_dir) / "metadata.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())["metrics"]


def is_better(challenger: dict[str, float], champion: dict[str, float] | None) -> bool:
    """Challenger wins if there is no champion or it improves the primary metric."""
    if champion is None:
        return True
    return challenger[PRIMARY_METRIC] > champion[PRIMARY_METRIC]


def stage_challenger(cfg: Config, synthetic: bool, shadow_dir: str | Path) -> dict[str, float]:
    from fraud_detection.train import train  # local import: heavy (mlflow, sklearn)

    metrics = train(cfg, synthetic=synthetic, out_dir=str(shadow_dir))
    logger.info("Challenger staged to %s", shadow_dir)
    return metrics


def promote(shadow_dir: str | Path, model_dir: str | Path) -> None:
    shadow_dir, model_dir = Path(shadow_dir), Path(model_dir)
    for name in ARTIFACTS:
        src = shadow_dir / name
        if src.exists():
            shutil.copy2(src, model_dir / name)
    logger.info("Promoted challenger from %s to %s", shadow_dir, model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--promote", action="store_true",
                        help="Promote the already-staged challenger (no retraining)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = Path(cfg.serving.model_dir)
    shadow_dir = model_dir / "shadow"

    champion = read_metrics(model_dir)

    if args.promote:
        challenger = read_metrics(shadow_dir)
        if challenger is None:
            logger.error("No staged challenger in %s - run retrain first", shadow_dir)
            sys.exit(2)
    else:
        challenger = stage_challenger(cfg, args.synthetic, shadow_dir)

    champ_str = f"{champion[PRIMARY_METRIC]:.4f}" if champion else "none"
    logger.info(
        "%s: challenger=%.4f champion=%s",
        PRIMARY_METRIC, challenger[PRIMARY_METRIC], champ_str,
    )

    if not is_better(challenger, champion):
        logger.warning("Challenger does NOT beat champion - not promoting")
        sys.exit(1)

    if args.promote:
        promote(shadow_dir, model_dir)
        print("Promoted. Restart the API (or redeploy) to serve the new champion.")
    else:
        print(
            "Challenger beats champion. Next: serve it in shadow mode\n"
            "  SHADOW_MODEL_DIR=models/shadow make serve\n"
            "then promote with:  python -m fraud_detection.retrain --promote"
        )


if __name__ == "__main__":
    main()
