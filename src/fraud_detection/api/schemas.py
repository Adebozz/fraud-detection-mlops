"""Request/response models for the prediction API."""

from pydantic import BaseModel, Field, create_model

from fraud_detection.data import FEATURE_COLUMNS

# Build the transaction schema dynamically from the canonical feature list
# so API and training can never drift apart.
Transaction = create_model(
    "Transaction",
    **{name: (float, Field(...)) for name in FEATURE_COLUMNS},
)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_run_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
