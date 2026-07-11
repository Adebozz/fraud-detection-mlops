.PHONY: install lint test train train-synthetic serve mlflow-ui docker-build docker-run \
        k3d-up k3d-deploy k3d-down eks-up eks-deploy eks-down

CLUSTER := fraud-detection

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest

train:
	python -m fraud_detection.train

train-synthetic:
	python -m fraud_detection.train --synthetic

serve:
	PRED_LOG_DIR=logs uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000 --reload

clean-logs:
	rm -rf logs

simulate:
	python scripts/simulate_traffic.py --n 300

simulate-drift:
	python scripts/simulate_traffic.py --n 300 --drift

drift-report:
	python -m fraud_detection.drift

retrain:
	python -m fraud_detection.retrain

retrain-if-drift:
	python -m fraud_detection.drift || python -m fraud_detection.retrain

serve-shadow:
	PRED_LOG_DIR=logs SHADOW_MODEL_DIR=models/shadow \
		uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000 --reload

promote:
	python -m fraud_detection.retrain --promote

demo:
	streamlit run app/demo.py

monitor-up:
	docker compose up -d prometheus grafana
	@echo "Grafana: http://localhost:3000 (dashboard: Fraud Detection API)"

monitor-down:
	docker compose down

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:
	docker build -t fraud-api .

docker-run:
	docker run -p 8000:8000 fraud-api

# ---------- Local Kubernetes (k3d - free) ----------

k3d-up:
	k3d cluster create $(CLUSTER) --agents 1

k3d-deploy: docker-build
	docker tag fraud-api fraud-api:local
	k3d image import fraud-api:local -c $(CLUSTER)
	kubectl apply -k k8s/overlays/local
	kubectl rollout restart deployment/fraud-api
	kubectl rollout status deployment/fraud-api --timeout=180s
	@echo "Run: kubectl port-forward svc/fraud-api 8000:80  ->  http://localhost:8000/docs"

k3d-down:
	k3d cluster delete $(CLUSTER)

# ---------- AWS EKS (costs money while running!) ----------

eks-up:
	cd terraform && terraform init && terraform apply
	aws eks update-kubeconfig --region eu-west-2 --name $(CLUSTER)

eks-deploy:
	kubectl apply -k k8s/overlays/aws
	kubectl rollout status deployment/fraud-api --timeout=180s
	kubectl get svc fraud-api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' && echo

eks-down:
	cd terraform && terraform destroy
