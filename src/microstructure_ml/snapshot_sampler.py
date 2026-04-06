from microstructure_ml.book_builder import BookBuilder

def take_snapshot(book: BookBuilder, timestamp: str, product: str, exchange: str, num_levels: int) -> dict:
    snapshot = {}
    top_bids = sorted(book.bids.keys(), reverse = True)[:num_levels]
    top_asks = sorted(book.asks.keys(), reverse = False)[:num_levels]
    
    num_active_bids = len(top_bids)
    i = 1
    while i <= num_active_bids:
        snapshot[f"bid_price_{i}"] = top_bids[i-1]
        snapshot[f"bid_size_{i}"] = book.bids[top_bids[i-1]]
        i += 1
    while i <= num_levels:
        snapshot[f"bid_price_{i}"] = None
        snapshot[f"bid_size_{i}"] = None
        i += 1
    
    num_active_asks = len(top_asks)
    i = 1
    while i <= num_active_asks:
        snapshot[f"ask_price_{i}"] = top_asks[i-1]
        snapshot[f"ask_size_{i}"] = book.asks[top_asks[i-1]]
        i += 1
    while i<= num_levels:
        snapshot[f"ask_price_{i}"] = None
        snapshot[f"ask_size_{i}"] = None
        i += 1
    
    snapshot["timestamp"] = timestamp
    snapshot["product"] = product
    snapshot["exchange"] = exchange

    return snapshot