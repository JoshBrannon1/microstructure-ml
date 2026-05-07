import pytest
import polars as pl
from src.microstructure_ml.label_builder import compute_labels

def test_compute_labels():
    df = pl.DataFrame({
        "mid_price": [50, 75, 100, 40]
    })
    result = compute_labels(df, [1])
    assert result["return_1s"].to_list()[:-1] == pytest.approx([0.5, 1/3, -0.6], rel=1e-4)
    assert result["return_1s"].to_list()[-1:] == [None]