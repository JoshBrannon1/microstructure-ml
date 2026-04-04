from websockets import connect
from typing import NamedTuple, Optional
import json

class BookUpdate(NamedTuple):
    side: str
    price: float
    size: float
    time: Optional[str]

class CoinbaseAdapter:
    def __init__(self, url = "wss://ws-feed.exchange.coinbase.com", product = "BTC-USD", channel = "level2"):
        self.url = url
        self.product = product
        self.channel = channel
        self.ws = None
    
    async def connect(self):
        self.ws = await connect(self.url)
        subscription = {"type": "subscribe", "channels": [self.channel], "product_ids": [self.product]}
        await self.ws.send(json.dumps(subscription))

    def parse_message(self, raw_json):
        message = json.loads(raw_json)
        book_updates = []
        
        #No explicit side for the snapshot message type, side is inferred from the dictionary index
        if message["type"] == "snapshot":
            bid_list = message["bids"]
            for bid in bid_list:
                book_updates.append(BookUpdate(side = "bid", price = float(bid[0]), size = float(bid[1]), time = None))
            
            ask_list = message["asks"]
            for ask in ask_list:
                book_updates.append(BookUpdate(side = "ask", price = float(ask[0]), size = float(ask[1]), time = None))
            
            return book_updates
        
        elif message["type"] == "l2update":
            changes_list = message["changes"]
            update_time = message["time"]
            for change in changes_list:
                book_updates.append(BookUpdate(side = change[0], price = float(change[1]), size = float(change[2]), time = update_time))

            return book_updates
        
        #Unrecognized messages are intentionally ignored
        else:
            return None
    
    async def listen(self):
        if self.ws is None:
            raise RuntimeError("Not connected, call connect() first")
        
        while True:
            message = await self.ws.recv()
            print(message)
            yield self.parse_message(message)

