import pytest
from microstructure_ml.book_builder import BookBuilder
from microstructure_ml.coinbase_adapter import BookUpdate
from microstructure_ml.snapshot_sampler import take_snapshot
from microstructure_ml.snapshot_writer import write_snapshots
from pathlib import Path
from datetime import datetime
import polars as pl

@pytest.fixture
def book():
    return BookBuilder()

def test_correct_shape(book, tmp_path):
    snapshot_updates_1 = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="bid", price=99.0, size=2.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None),
        BookUpdate(side="ask", price=102.0, size=3.0, time=None)
    ]
    snapshot_updates_2 = [
        BookUpdate(side="bid", price=99.0, size=1.0, time=None),
        BookUpdate(side="bid", price=98.0, size=2.0, time=None),
        BookUpdate(side="ask", price=100.0, size=1.5, time=None),
    ]

    book.apply_snapshot(snapshot_updates_1)
    snapshot1 = take_snapshot(book, "2024-01-01T00:00:00Z", "BTC-USD", "Kraken", 2)
    book.apply_snapshot(snapshot_updates_2)
    snapshot2 = take_snapshot(book, "2024-01-01T00:10:00Z", "BTC-USD", "Kraken", 2)
    writer = write_snapshots([snapshot1, snapshot2], "BTC/USD", "Kraken", tmp_path)
    date_str = datetime.now().strftime("%Y-%m-%d")
    expected_path = Path(tmp_path) / f"exchange=Kraken" / f"product=BTC-USD" / f"date={date_str}"
    assert expected_path.exists()
    assert any(expected_path.iterdir())
    assert len(list(expected_path.iterdir())) == 1
    df = pl.read_parquet(next(expected_path.iterdir()))
    assert df.shape == (2, 11)
    

