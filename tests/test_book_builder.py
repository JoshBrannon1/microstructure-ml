import pytest
from microstructure_ml.book_builder import BookBuilder
from microstructure_ml.coinbase_adapter import BookUpdate

@pytest.fixture
def book():
    return BookBuilder()

def test_apply_snapshot(book):
    snapshot_updates = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="bid", price=99.0, size=2.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None),
        BookUpdate(side="ask", price=102.0, size=3.0, time=None)
    ]
    book.apply_snapshot(snapshot_updates)
    assert book.bids == {100.0: 1.0, 99.0: 2.0}
    assert book.asks == {101.0: 1.5, 102.0: 3.0}
    assert book.best_bid == 100.0
    assert book.best_ask == 101.0

def test_apply_update(book):
    snapshot_updates = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="bid", price=99.0, size=2.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None),
        BookUpdate(side="ask", price=102.0, size=3.0, time=None)
    ]
    book.apply_snapshot(snapshot_updates)

    update_updates = [
        BookUpdate(side="bid", price=100.0, size=0.5, time=None),  # Update existing bid
        BookUpdate(side="bid", price=98.0, size=1.0, time=None),   # Add new bid
        BookUpdate(side="ask", price=101.0, size=0.0, time=None),  # Remove existing ask
        BookUpdate(side="ask", price=103.0, size=2.0, time=None)   # Add new ask
    ]
    book.apply_update(update_updates)
    assert book.bids == {100.0: 0.5, 99.0: 2.0, 98.0: 1.0}
    assert book.asks == {102.0: 3.0, 103.0: 2.0}
    assert book.best_bid == 100.0
    assert book.best_ask == 102.0

def test_best_price_validation(book):
    snapshot_updates = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None)
    ]
    book.apply_snapshot(snapshot_updates)

    # This update should raise a ValueError because it would make the best bid greater than or equal to the best ask
    invalid_update = [
        BookUpdate(side="bid", price=102.0, size=1.0, time=None)
    ]
    with pytest.raises(ValueError):
        book.apply_update(invalid_update)

def test_size_zero(book):
    snapshot_updates = [
        BookUpdate(side="bid", price=100.0, size=1.0, time=None),
        BookUpdate(side="ask", price=101.0, size=1.5, time=None)
    ]
    book.apply_snapshot(snapshot_updates)

    # This update should remove the existing bid at 100.0
    zero_size_update = [
        BookUpdate(side="bid", price=100.0, size=0.0, time=None)
    ]
    book.apply_update(zero_size_update)
    assert book.bids == {}
    assert book.best_bid is None
    assert book.asks == {101.0: 1.5}
    assert book.best_ask == 101.0