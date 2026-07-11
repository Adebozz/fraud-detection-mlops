.PHONY: install lint test train train-synthetic serve mlflow-ui docker-build docker-run

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
	uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:
	docker build -t fraud-api .

docker-run:
	docker run -p 8000:8000 fraud-api
