import microstructure_ml.training_pipeline as tp
from sklearn.linear_model import LinearRegression
import pytest
import polars as pl
from pathlib import Path
from sklearn.utils.validation import check_is_fitted

def test_load_split(tmp_path):
    # Create a temporary directory with parquet files
    dir1 = tmp_path / "date=2023-01-01"
    dir1.mkdir()
    df1 = pl.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0],
        "label": [2, 3]
    })
    df1.write_parquet(dir1 / "data.parquet")

    x, y = tp.load_split([dir1], ["feature1", "feature2"], "label")
    assert x.shape == (2, 2)
    assert y.shape == (2,)
    assert (x[:, 0] == [1.0, 2.0]).all()
    assert (x[:, 1] == [3.0, 4.0]).all()
    assert (y == [2, 3]).all()

def test_train_model():
    x_train = [[1.0, 3.0], [2.0, 4.0]]
    y_train = [2, 3]
    model = tp.train_model(x_train, y_train, tp.LinearRegression)
    assert model.coef_.shape == (2,)
    check_is_fitted(model)

def test_evaluate_model():
    x_val = [[1.0, 3.0], [2.0, 4.0]]
    y_val = [2, 3]
    model = LinearRegression().fit(x_val, y_val)
    mse = tp.evaluate_model(model, x_val, y_val)
    assert mse == 0.0
