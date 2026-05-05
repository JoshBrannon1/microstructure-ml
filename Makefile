.PHONY: install collect snapshots features dataset train eval test

OUTPUT ?= data/features

help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  collect    - Collect raw market data"
	@echo "  snapshots  - Process raw data into snapshots"
	@echo "  features   - Engineer features from snapshots"
	@echo "  dataset    - Build final dataset"
	@echo "  train      - Train model"
	@echo "  eval       - Evaluate model"
	@echo "  test       - Run tests"

install:
	 python3 -m poetry install

collect:
	poetry run python -m microstructure_ml.collector

snapshots:
	poetry run python -m microstructure_ml.snapshots

features:
	poetry run python -m microstructure_ml.feature_pipeline --input $(INPUT) --output $(OUTPUT)

dataset:
	poetry run python -m microstructure_ml.dataset

train:
	poetry run python -m microstructure_ml.train

eval:
	poetry run python -m microstructure_ml.eval

test:
	poetry run python -m microstructure_ml.feature_pipeline --input $(INPUT) --output $(OUTPUT) -m pytest tests/ -v
