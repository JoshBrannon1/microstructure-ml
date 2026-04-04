from microstructure_ml.coinbase_adapter import BookUpdate
from websockets import connect
import json

class KrakenAdapter:
    def __init__(self, url = "wss://ws.kraken.com/v2", symbol = "BTC/USD", channel = "book"):
        self.url = url
        self.symbol = symbol
        self.channel = channel
        self.ws = None
    
    async def connect(self):
        self.ws = await connect(self.url)
        subscribe_message = {"method": "subscribe", "params": {"channel": self.channel, "symbol": [self.symbol]}}
        await self.ws.send(json.dumps(subscribe_message))
    
    async def reconnect(self):
        try:
            await self.ws.close()
        except Exception:
            pass #connection could already be broken
        self.ws = None
        await self.connect()
    
    def parse_message(self, raw_message):
        message = json.loads(raw_message)
        book_updates = []

        if message.get("channel") != "book":
            return None
        
        #only access data[0] since we are only subscribed to one symbol, and the data is always in the first element of the data array
        if message["type"] == "snapshot":
            bid_list = message["data"][0]["bids"]
            for bid in bid_list:
                book_updates.append(BookUpdate(side="bid", price=float(bid["price"]), size=float(bid["qty"]), time=message["data"][0]["timestamp"]))

            ask_list = message["data"][0]["asks"]
            for ask in ask_list:
                book_updates.append(BookUpdate(side="ask", price=float(ask["price"]), size=float(ask["qty"]), time=message["data"][0]["timestamp"]))
            
            return book_updates, message["type"]
        
        elif message["type"] == "update":
            bid_list = message["data"][0]["bids"]
            for bid in bid_list:
                book_updates.append(BookUpdate(side = "bid", price=float(bid["price"]), size=float(bid["qty"]), time = message["data"][0]["timestamp"]))

            ask_list = message["data"][0]["asks"]
            for ask in ask_list:
                book_updates.append(BookUpdate(side = "ask", price=float(ask["price"]), size=float(ask["qty"]), time = message["data"][0]["timestamp"]))
            
            return book_updates, message["type"]
        
        #Unrecognized messages are intentionally ignored
        else:
            return None
    
    async def listen(self):
        if self.ws is None:
            raise RuntimeError("Not connected, call connect() first")
        
        while True:
            message = await self.ws.recv()
            result = self.parse_message(message)
            if result is not None:
                yield result