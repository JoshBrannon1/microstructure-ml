## Microstructure ML research platform

## Motivation 
Many price prediction projects use OHLCV data: open, high, low, close, volume. The problem is that OHLCV is a lossy compression of thousands of individual market events into five numbers. The order book is the raw signal underneath. It tells you not just what price traded, but the full structure of supply and demand, including: how much volume is sitting at each price level, how deep you need to eat into the asks to fill a position, and whether buyers or sellers are dominating in real time. Order flow imbalance (the imbalance between bid and ask volume at the top of the book) has well-documented predictive power at short horizons; this project aims to be built around that signal. Something important to note is that ML projects using this kind of data often get the evaluation wrong (for example: training and testing on overlapping time windows inflates results). Getting my methodology right is as much the point as the model itself.

## Research Question 
How much short-horizon predictive signal exists in top-of-book microstructure features for BTC-USD, and how stable is it out-of-sample under leakage-aware validation?

## Quickstart
```
git clone https://github.com/JoshBrannon1/microstructure-ml.git
make install
make collect
make dataset
make train
make eval
```

## Project Structure
```
microstructure-ml/
├── src/          # Core package: data ingestion, feature engineering, modeling, and evaluation 
├── tests/        # Unit tests for book, features, splits
├── configs/      # All run parameters in one place
├── data/         # Book snapshots, features, final training datasets
└── reports/      # Plots and writeup
```
## Status

| Week | Objective | Status |
|------|-----------|--------|
| 1 | Repo scaffolding + environment | Done |
| 1 | Exchange adapter (Kraken) | Done |
| 1 | Local L2 book builder + storage | Done |
| 1 | Correctness validation + resync | Done |
| 2 | Snapshot sampler + Parquet storage | Up next |
| 3 | Features (microprice/imbalance/spread/depth) | Planned |
| 4 | Join labels and dataset | Planned |
| 5 | Leakage-resistant validation | Planned |
| 6 | Baseline models + interpretation | Planned |
| 7 | Execution simulator + reporting | Planned |
