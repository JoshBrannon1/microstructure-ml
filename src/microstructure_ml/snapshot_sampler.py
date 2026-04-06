from microstructure_ml.book_builder import BookBuilder
from microstructure_ml.collector import Collector

def take_snapshot(book: BookBuilder, timestamp: str, product: str, exchange: str, num_levels: int) -> dict:
    snapshot = {}
    top_bids = sorted(book.bids.keys(), reverse = True)[:10]
    top_asks = sorted(book.asks.keys(), reverse = False)[:10]
    num_active_bid_levels = len(top_bids)
    
    i = 1
    while i <= num_active_bid_levels:
        snapshot[f"bid_price_{i}"] = top_bids[i]
        snapshot[f"bid_size_{i}"] = book.bids[top_bids[i]]
        i += 1
    while i <= num_levels:
        snapshot[f"bid_price{i}"] = None
        snapshot[f"bid_size{i}"] = None
    
    num_active_ask_levels = len(top_asks)
    i = 1
    for ask in bottom_asks:
        snapshot[f"ask_price_{i}"] = ask
        snapshot[f"ask_size_{i}"] = book.asks[ask]
        i += 1
    snapshot["timestamp"] = timestamp
    snapshot["product"] = product
    snapshot["exchange"] = exchange



    
    