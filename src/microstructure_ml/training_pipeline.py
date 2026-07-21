import polars as pl
import argparse
from pathlib import Path
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from microstructure_ml.validation_split import create_splits, list_dates
from sklearn.ensemble import RandomForestRegressor

MODELS = {
        "linear": LinearRegression,
        "rf": RandomForestRegressor,
    }

def load_split(data: list[Path], keep_columns: list[str], label: str) -> tuple[NDArray, NDArray]:
    files = []
    frames = []
    for file in data:
        files.extend(sorted(file.glob("*.parquet")))
    for file in files:
        frames.append(pl.read_parquet(file).select(keep_columns + [label]))
    if not frames:
        raise ValueError(f"No parquet files found in {data}")
    df = pl.concat(frames)
    df = df.drop_nulls()
    if df.is_empty():
        raise ValueError(f"No data remaining after dropping nulls for dates: {data}")
    return df.drop(label).to_numpy(), df.get_column(label).to_numpy()

def train_model(x_train: NDArray, y_train: NDArray, model_class: type, **kwargs) -> BaseEstimator:
    model = model_class(**kwargs)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model: BaseEstimator, x_val, y_val) -> float:
    return mean_squared_error(y_val, model.predict(x_val))

def run_walk_forward(train_val_splits: list, keep_columns: list[str], label: str, model_class: type, **kwargs) -> list[float]:
    val_scores = []
    for i, (train_dates, val_dates) in enumerate(train_val_splits):
        print(f"Running split {i+1}/{len(train_val_splits)}")
        x_train, y_train = load_split(train_dates, keep_columns, label)
        x_val, y_val = load_split(val_dates, keep_columns, label)
        model = train_model(x_train, y_train, model_class, **kwargs)
        val_score = evaluate_model(model, x_val, y_val)
        print(f"Validation MSE for split {i+1}: {val_score}")
        val_scores.append(val_score)
    return val_scores

def entry_point(input_base: Path, val_days: int, step_days: int, test_days: int, keep_columns: list[str], label: str, model_class: str, **kwargs):
    train_val_splits, test_dates = create_splits(input_base, val_days, step_days, test_days)
    val_scores = run_walk_forward(train_val_splits, keep_columns, label, MODELS[model_class], **kwargs)
    print(f"Average validation MSE across splits: {sum(val_scores) / len(val_scores)}")

    all_dates = list_dates(input_base)
    train_all = [d for d in all_dates if d not in test_dates]
    full_model = train_model(*load_split(train_all, keep_columns, label), MODELS[model_class], **kwargs)
    x_test, y_test = load_split(test_dates, keep_columns, label)
    test_score = evaluate_model(full_model, x_test, y_test)
    print(f"Test MSE: {test_score}")

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Base directory for input data")
    parser.add_argument("--val_days", type=int, default=5, help="Number of days to use for validation in each split")
    parser.add_argument("--step_days", type=int, default=5, help="Number of days to step forward for each split")
    parser.add_argument("--test_days", type=int, default=10, help="Number of days to hold out for final testing")
    parser.add_argument("--keep_columns", nargs="+", required=True, help="Columns to keep as features")
    parser.add_argument("--label", type=str, required=True, help="Column to use as label")
    parser.add_argument("--model_class", type=str, choices=["linear", "rf"], default="linear", help="Model class to use for training")
    args = parser.parse_args()
    entry_point(args.input, args.val_days, args.step_days, args.test_days, args.keep_columns, args.label, args.model_class)