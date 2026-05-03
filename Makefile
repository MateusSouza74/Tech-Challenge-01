.PHONY: install lint test run train mlflow

install:
	pip install -e ".[dev]"

train:
	python -m src.training.train

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v

run:
	uvicorn src.api.api:app --reload

mlflow:
	mlflow ui
