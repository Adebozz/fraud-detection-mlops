import json
from pathlib import Path

from fraud_detection.retrain import is_better, promote, read_metrics


def _write_model(dir_: Path, pr_auc: float) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "model.json").write_text('{"fake": "model"}')
    (dir_ / "metadata.json").write_text(
        json.dumps({"run_id": f"run-{pr_auc}", "metrics": {"pr_auc": pr_auc}})
    )


def test_is_better_no_champion():
    assert is_better({"pr_auc": 0.5}, None)


def test_is_better_comparison():
    assert is_better({"pr_auc": 0.90}, {"pr_auc": 0.88})
    assert not is_better({"pr_auc": 0.88}, {"pr_auc": 0.90})
    assert not is_better({"pr_auc": 0.90}, {"pr_auc": 0.90})  # ties don't promote


def test_read_metrics_missing_dir(tmp_path):
    assert read_metrics(tmp_path / "nope") is None


def test_promote_copies_challenger(tmp_path):
    champion_dir, shadow_dir = tmp_path / "models", tmp_path / "models" / "shadow"
    _write_model(champion_dir, pr_auc=0.85)
    _write_model(shadow_dir, pr_auc=0.91)

    promote(shadow_dir, champion_dir)

    promoted = read_metrics(champion_dir)
    assert promoted["pr_auc"] == 0.91
    # challenger staging area still intact (promotion copies, not moves)
    assert read_metrics(shadow_dir)["pr_auc"] == 0.91
