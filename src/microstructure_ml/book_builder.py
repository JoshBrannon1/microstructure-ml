from microstructure_ml.coinbase_adapter import BookUpdate

class BookBuilder:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.best_bid = None
        self.best_ask = None
        self.is_valid = True

    def apply_snapshot(self, book_updates):
        self.bids = {}
        self.asks = {}
        for update in book_updates:
            if update.side == "bid":
                self.bids[update.price] = update.size
            elif update.side == "ask":
                self.asks[update.price] = update.size
        
        self.is_valid = True
        self.update_best_prices()
        
    def apply_update(self, book_updates):
        if (not self.is_valid):
            return
        
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
            self.is_valid = False
            print(f"Invariant violated: best_bid={self.best_bid} >= best_ask={self.best_ask}")
    
    def reset(self):
        self.bids = {}
        self.asks = {}
        self.best_bid = None
        self.best_ask = None
        self.is_valid = False