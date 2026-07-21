.PHONY: install collect snapshots features labels dataset train eval test

FEATURES_OUTPUT ?= data/features
LABELS_OUTPUT ?= data/labels
INTERVALS ?= 5 10 30
VAL_DAYS ?= 3
STEP_DAYS ?= 1
TEST_DAYS ?= 7
KEEP_COLUMNS ?= mid_price spread imbalance microprice depth_imbalance
LABEL ?= microprice_dev_5s
MODEL_CLASS ?= linear

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

train:
	poetry run python -m microstructure_ml.training_pipeline --input $(INPUT) --val_days $(VAL_DAYS) --step_days $(STEP_DAYS) --test_days $(TEST_DAYS) --keep_columns $(KEEP_COLUMNS) --label $(LABEL) --model_class $(MODEL_CLASS)

eval:
	poetry run python -m microstructure_ml.eval

test:
	poetry run python -m pytest tests/ -v
