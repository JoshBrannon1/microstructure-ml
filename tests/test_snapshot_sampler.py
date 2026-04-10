import pytest
from microstructure_ml.snapshot_sampler import take_snapshot
from microstructure_ml.book_builder import BookBuilder
from microstructure_ml.coinbase_adapter import BookUpdate

@pytest.fixture
def book():
    return BookBuilder()

def test_correct_top_bids_asks(book):
    snapshot_updates = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="bid", price=99.0, size=2.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None),
        BookUpdate(side="ask", price=102.0, size=3.0, time=None)
    ]
    book.apply_snapshot(snapshot_updates)
    snapshot = take_snapshot(book, "2024-01-01T00:00:00Z", "BTC-USD", "Kraken", 2)
    assert snapshot["bid_price_1"] == 100.0
    assert snapshot["bid_size_1"] == 1.0
    assert snapshot["bid_price_2"] == 99.0
    assert snapshot["bid_size_2"] == 2.0
    assert snapshot["ask_price_1"] == 101.0
    assert snapshot["ask_size_1"] == 1.5
    assert snapshot["ask_price_2"] == 102.0
    assert snapshot["ask_size_2"] == 3.0

def test_fewer_levels_than_requested(book):
    snapshot_updates = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None)
    ]
    book.apply_snapshot(snapshot_updates)
    snapshot = take_snapshot(book, "2024-01-01T00:00:00Z", "BTC-USD", "Kraken", 3)
    assert snapshot["bid_price_1"] == 100.0
    assert snapshot["bid_size_1"] == 1.0
    assert snapshot["bid_price_2"] == None
    assert snapshot["bid_size_2"] == None
    assert snapshot["bid_price_3"] == None
    assert snapshot["bid_size_3"] == None
    assert snapshot["ask_price_1"] == 101.0
    assert snapshot["ask_size_1"] == 1.5
    assert snapshot["ask_price_2"] == None
    assert snapshot["ask_size_2"] == None
    assert snapshot["ask_price_3"] == None
    assert snapshot["ask_size_3"] == None

def test_empty_book(book):
    snapshot = take_snapshot(book, "2024-01-01T00:00:00Z", "BTC-USD", "Kraken", 2)
    assert snapshot["bid_price_1"] == None
    assert snapshot["bid_size_1"] == None
    assert snapshot["bid_price_2"] == None
    assert snapshot["bid_size_2"] == None
    assert snapshot["ask_price_1"] == None
    assert snapshot["ask_size_1"] == None
    assert snapshot["ask_price_2"] == None
    assert snapshot["ask_size_2"] == None

def test_correct_key_count(book):
    snapshot_updates = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="bid", price=99.0, size=2.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None),
        BookUpdate(side="ask", price=102.0, size=3.0, time=None)
    ]
    book.apply_snapshot(snapshot_updates)
    snapshot = take_snapshot(book, "2024-01-01T00:00:00Z", "BTC-USD", "Kraken", 2)
    assert len(snapshot) == 11 #4 price/size pairs + timestamp + product + exchange

def test_ordering(book):
    snapshot_updates = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="bid", price=99.0, size=2.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None),
        BookUpdate(side="ask", price=102.0, size=3.0, time=None)
    ]
    book.apply_snapshot(snapshot_updates)
    snapshot = take_snapshot(book, "2024-01-01T00:00:00Z", "BTC-USD", "Kraken", 2)
    assert snapshot["bid_price_1"] > snapshot["bid_price_2"]
    assert snapshot["ask_price_1"] < snapshot["ask_price_2"]