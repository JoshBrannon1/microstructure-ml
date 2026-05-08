from pathlib import Path
from unittest.mock import patch
import pytest
from microstructure_ml.validation_split import train_val_splits, create_splits

def test_correct_split():
    dates = [1, 2, 3, 4, 5, 6]
    assert train_val_splits(dates, 1, 1) == [([1],[2]), ([1,2],[3]), ([1,2,3],[4]), ([1,2,3,4],[5]), ([1,2,3,4,5],[6])]

def test_validation_cutoff():
    dates = [1, 2, 3, 4, 5, 6]
    assert train_val_splits(dates, 2, 1) == [([1],[2,3]), ([1,2],[3,4]), ([1,2,3],[4,5]), ([1,2,3,4],[5,6])]

def test_insufficient_data_raises():
    with patch("microstructure_ml.validation_split.list_dates", return_value=[1, 2, 3]):
        with pytest.raises(ValueError):
            create_splits(Path("fake/path"), val_days=2, step_days=1, test_days=3)
