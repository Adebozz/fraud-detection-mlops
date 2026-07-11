"""Typed access to configs/config.yaml."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@dataclass
class DataConfig:
    openml_id: int = 1597
    raw_path: str = "data/raw/creditcard.parquet"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    type: str = "xgboost"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MlflowConfig:
    tracking_uri: str = "mlruns"
    experiment_name: str = "fraud-detection"
    registered_model_name: str = "fraud-xgb"


@dataclass
class ServingConfig:
    model_dir: str = "models"
    decision_threshold: float = 0.5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        mlflow=MlflowConfig(**raw.get("mlflow", {})),
        serving=ServingConfig(**raw.get("serving", {})),
    )
