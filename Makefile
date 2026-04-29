.PHONY: install lint test run

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v

run:
	uvicorn src.api.api:app --reload
