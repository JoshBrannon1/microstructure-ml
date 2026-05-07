.PHONY: install collect snapshots features dataset train eval test

FEATURES_OUTPUT ?= data/features
LABELS_OUTPUT ?= data/labels
INTERVALS ?= 5 10 30

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
	 poetry install

collect:
	poetry run python -m microstructure_ml.collector

snapshots:
	poetry run python -m microstructure_ml.snapshots

features:
	poetry run python -m microstructure_ml.feature_pipeline --input $(INPUT) --output $(FEATURES_OUTPUT)

labels:
	poetry run python -m microstructure_ml.label_builder --input $(INPUT) --output $(LABELS_OUTPUT) --intervals $(INTERVALS)

dataset:
	poetry run python -m microstructure_ml.dataset

train:
	poetry run python -m microstructure_ml.train

eval:
	poetry run python -m microstructure_ml.eval

test:
	poetry run python -m pytest tests/ -v
