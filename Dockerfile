# Serving image: FastAPI + exported XGBoost model.
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install only serving dependencies (no mlflow/sklearn needed at inference).
RUN pip install --no-cache-dir \
    "fastapi>=0.111" "uvicorn[standard]>=0.30" "pydantic>=2.7" \
    "xgboost>=2.0" "numpy>=1.26" "PyYAML>=6.0"

COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e . 2>/dev/null || true
ENV PYTHONPATH=/app/src

# Model artefacts are baked in for simplicity; mount or fetch from a
# registry (S3/MLflow) in production instead.
COPY models/ models/
ENV MODEL_DIR=/app/models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "fraud_detection.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
