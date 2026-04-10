from microstructure_ml.coinbase_adapter import BookUpdate
from collections import deque
from typing import NamedTuple, Optional

class BookStatus(NamedTuple):
    is_valid: bool
    reason: Optional[str]
    is_anomalous_spread: bool

class BookBuilder:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.best_bid = None
        self.best_ask = None
        self.status = BookStatus(is_valid = False, reason = "Book not initialized, waiting for first snapshot", is_anomalous_spread = False)
        self.spread_history = deque(maxlen=10)
        self.min_spread_observations = 20

    def apply_snapshot(self, book_updates):
        self.bids = {}
        self.asks = {}
        for update in book_updates:
            if update.side == "bid":
                self.bids[update.price] = update.size
            elif update.side == "ask":
                self.asks[update.price] = update.size
        
        self.status = BookStatus(is_valid = True, reason = None, is_anomalous_spread = False)
        self.update_best_prices()
        self.spread_history.append(self.best_ask - self.best_bid) if self.best_bid is not None and self.best_ask is not None else None

        
    def apply_update(self, book_updates):
        if (not self.status.is_valid):
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
            self.status = BookStatus(is_valid = False, reason = f"Invariant violated: best_bid={self.best_bid} >= best_ask={self.best_ask}", is_anomalous_spread = False)
            print(self.status.reason)
    
    def reset(self):
        self.bids = {}
        self.asks = {}
        self.best_bid = None
        self.best_ask = None
        self.status = BookStatus(is_valid = False, reason = "Book reset, waiting for next snapshot", is_anomalous_spread = False)
        self.spread_history = deque(maxlen=10)
    
    def health_check(self):
        if self.status.is_valid and self.best_bid is not None and self.best_ask is not None:
            current_spread = (self.best_ask - self.best_bid)
            if len(self.spread_history) < self.min_spread_observations:
                self.spread_history.append(current_spread)
            else:
                average_spread = sum(self.spread_history) / len(self.spread_history)
                if current_spread > 2 * average_spread:
                    print(f"Unusual spread detected: {current_spread} > 2 * {average_spread}")
                    self.status = BookStatus(is_valid = True, reason = f"Unusual spread detected: {current_spread} > 2 * {average_spread}", is_anomalous_spread = True)
                else:
                    self.status = BookStatus(is_valid = True, reason = None, is_anomalous_spread = False)
                
                if not self.status.is_anomalous_spread:
                    self.spread_history.append(current_spread)