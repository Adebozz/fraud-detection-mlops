# Fraud Detection — End-to-End MLOps Pipeline

Production-grade credit-card fraud detection: reproducible training pipeline with MLflow
experiment tracking, exported model served via FastAPI, containerised with Docker, and
CI/CD through GitHub Actions publishing to GHCR.

> Portfolio project — phase 1 of 4 (training + serving + CI). Phases 2–4 add
> Kubernetes deployment, monitoring/drift detection, and automated retraining.

## Architecture

```
OpenML dataset ──> data.py (download, cache, validate)
                        │
                        ▼
                 train.py ──────────────┐
                 XGBoost + class        │ logs params/metrics/model
                 imbalance weighting    ▼
                        │            MLflow (tracking + registry)
                        ▼
                 models/model.json + metadata.json   (exported artefact)
                        │
                        ▼
                 FastAPI  /predict  /health  /model-info
                        │
                        ▼
                 Docker image ──> GHCR (via GitHub Actions)
```

**Dataset:** [ULB credit-card fraud](https://www.openml.org/d/1597) — 284,807 transactions,
0.17% fraud. A schema-identical **synthetic generator** powers tests and CI so the
pipeline runs anywhere without the 150 MB download.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install

make train              # real data (downloads once, ~150 MB) — or: make train-synthetic
make mlflow-ui          # browse experiments at http://localhost:5000
make serve              # API at http://localhost:8000/docs
make test               # pytest suite
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "$(python -c 'import json; from fraud_detection.data import FEATURE_COLUMNS; print(json.dumps({f: 0.0 for f in FEATURE_COLUMNS}))')"
```

## Results

| Metric | Value |
|---|---|
| ROC-AUC | _run `make train` and fill in_ |
| PR-AUC | _…_ |
| Recall @ 0.5 | _…_ |
| Precision @ 0.5 | _…_ |

PR-AUC is the headline metric: with 0.17% positives, ROC-AUC is inflated and
precision/recall trade-offs are what a fraud team actually operates on.

## Design decisions

- **Class imbalance** handled with `scale_pos_weight` rather than SMOTE — cheaper,
  no synthetic leakage, and well-suited to gradient-boosted trees.
- **Serving decoupled from training**: the API loads a plain XGBoost JSON artefact +
  metadata, so the inference image doesn't need sklearn/MLflow (smaller, faster cold start).
- **Schema single-source-of-truth**: API request models are generated from the same
  `FEATURE_COLUMNS` list used in training — schema drift between train and serve is impossible.
- **Synthetic data path** (`--synthetic`) lets CI smoke-train the full pipeline in seconds.

## Roadmap

- [x] Phase 1 — training pipeline, MLflow tracking, FastAPI serving, Docker, CI
- [ ] Phase 2 — deploy to Kubernetes (k3s → AWS EKS), Terraform IaC
- [ ] Phase 3 — monitoring: Evidently drift detection + Grafana dashboard
- [ ] Phase 4 — automated retraining triggers + shadow deployment
# fraud-detection-mlops
