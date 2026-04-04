from microstructure_ml.coinbase_adapter import BookUpdate

class BookBuilder:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.best_bid = None
        self.best_ask = None

    def apply_snapshot(self, book_updates):
        self.bids = {}
        self.asks = {}
        for update in book_updates:
            if update.side == "bid":
                self.bids[update.price] = update.size
            elif update.side == "ask":
                self.asks[update.price] = update.size
        
        self.update_best_prices()

    def apply_update(self, book_updates):
        for update in book_updates:
            if update.side == "bid":
                if update.size == 0:
                    self.bids.pop(update.price, None)
                else:
                    self.bids[update.price] = update.size
            elif update.side == "ask":
                if update.size == 0:
                    self.asks.pop(update.price, None)
                else:
                    self.asks[update.price] = update.size
        
        self.update_best_prices()

    def update_best_prices(self):
        self.best_bid = max(self.bids.keys()) if self.bids else None
        self.best_ask = min(self.asks.keys()) if self.asks else None
        if self.best_bid is not None and self.best_ask is not None and self.best_bid >= self.best_ask:
            raise ValueError("Best bid cannot be greater than or equal to best ask")
    