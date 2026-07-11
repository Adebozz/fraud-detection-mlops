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

Held-out test set (20%, stratified), 284,807 transactions, 0.17% fraud:

| Metric | Value |
|---|---|
| ROC-AUC | 0.973 |
| PR-AUC | 0.880 |
| Precision @ 0.5 | 0.882 |
| Recall @ 0.5 | 0.837 |

At the default threshold the model catches ~84% of fraud while ~88% of its
alerts are true fraud — i.e. roughly 1 false alarm per 7 flagged transactions
on a base rate of 1-in-580.

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

## Kubernetes deployment

Manifests live in `k8s/` (Kustomize: shared `base`, `local`/`aws` overlays);
infra in `terraform/` (VPC + EKS, eu-west-2).

**Local (free), needs Docker + k3d + kubectl:**

```bash
make k3d-up          # create local cluster
make k3d-deploy      # build image, import, deploy, wait for rollout
kubectl port-forward svc/fraud-api 8000:80   # then open http://localhost:8000/docs
make k3d-down        # tear down
```

**AWS EKS (~£3-4/day while running — always destroy after demos):**

```bash
make eks-up          # terraform apply (~15 min) + kubeconfig
make eks-deploy      # deploy GHCR image, prints the load-balancer URL
make eks-down        # terraform destroy - do not skip this
```

Production touches baked in: liveness/readiness probes on `/health`, resource
requests/limits, HPA (2-5 replicas at 70% CPU, backed by metrics-server),
single-NAT VPC to halve networking cost.

## Roadmap

- [x] Phase 1 — training pipeline, MLflow tracking, FastAPI serving, Docker, CI
- [x] Phase 2 — Kubernetes (k3d + AWS EKS), Terraform IaC
- [ ] Phase 3 — monitoring: Evidently drift detection + Grafana dashboard
- [ ] Phase 4 — automated retraining triggers + shadow deployment
# fraud-detection-mlops
