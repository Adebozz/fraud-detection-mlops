"""FastAPI service serving the exported XGBoost model.

Run locally:
    uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000

The model directory is resolved from $MODEL_DIR (default: models/), which
must contain model.json and metadata.json produced by the training pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import time

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from prometheus_client import make_asgi_app

from fraud_detection.api import monitoring
from fraud_detection.api.schemas import HealthResponse, PredictionResponse, Transaction

logger = logging.getLogger(__name__)

_state: dict = {"booster": None, "metadata": None, "shadow": None}


def _load_booster(model_dir: Path) -> tuple[xgb.Booster, dict]:
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))
    metadata = json.loads((model_dir / "metadata.json").read_text())
    return booster, metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", "models"))
    try:
        _state["booster"], _state["metadata"] = _load_booster(model_dir)
        logger.info("Loaded champion run %s from %s", _state["metadata"].get("run_id"), model_dir)
    except FileNotFoundError:
        logger.warning("No model found in %s - /predict will return 503", model_dir)

    shadow_dir = os.environ.get("SHADOW_MODEL_DIR")
    if shadow_dir:
        try:
            _state["shadow"], shadow_meta = _load_booster(Path(shadow_dir))
            logger.info("Shadow mode ON: challenger run %s", shadow_meta.get("run_id"))
        except FileNotFoundError:
            logger.warning("SHADOW_MODEL_DIR=%s set but no model found", shadow_dir)

    yield
    _state["booster"] = _state["shadow"] = None


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_state["booster"] is not None)


@app.get("/model-info")
def model_info() -> dict:
    if _state["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _state["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction) -> PredictionResponse:
    if _state["booster"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    meta = _state["metadata"]
    features = meta["features"]
    start = time.perf_counter()
    row = np.array([[getattr(txn, f) for f in features]], dtype=np.float32)
    dmatrix = xgb.DMatrix(row, feature_names=features)
    proba = float(_state["booster"].predict(dmatrix)[0])
    latency = time.perf_counter() - start
    threshold = float(meta["threshold"])

    monitoring.observe(proba, proba >= threshold, latency)

    # Shadow (challenger) scores the same request; response stays champion-only.
    shadow_proba = None
    if _state["shadow"] is not None:
        shadow_proba = float(_state["shadow"].predict(dmatrix)[0])
        monitoring.observe_shadow(shadow_proba, proba >= threshold, shadow_proba >= threshold)

    monitoring.log_prediction(
        {f: getattr(txn, f) for f in features}, proba, proba >= threshold, shadow_proba
    )

    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=proba >= threshold,
        threshold=threshold,
        model_run_id=meta.get("run_id", "unknown"),
    )
