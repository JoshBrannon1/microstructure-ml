from microstructure_ml.kraken_adapter import KrakenAdapter
from microstructure_ml.book_builder import BookBuilder
from microstructure_ml.snapshot_sampler import take_snapshot
from microstructure_ml.snapshot_writer import write_snapshots   
import asyncio
import datetime

class Collector:
    def __init__(self):
        self.adapter = KrakenAdapter()
        self.book_builder = BookBuilder()
        self.buffer = []
    
    async def run(self):
        await self.adapter.connect()
        asyncio.create_task(self.sample_loop("BTC/USD", "Kraken", 10, 1))
        async for message in self.adapter.listen():
            if message is not None:
                book_updates, update_type = message
                if update_type == "snapshot":
                    self.book_builder.apply_snapshot(book_updates)
                elif update_type == "update":
                    self.book_builder.apply_update(book_updates)

                if not self.book_builder.status.is_valid:
                    print(f"Book invalid, reason: {self.book_builder.status.reason}")
                    self.book_builder.reset()
                    await self.adapter.reconnect()
        
    async def sample_loop(self, product: str, exchange: str, num_levels, sample_interval: int):
        start_time = datetime.datetime.now()
        while self.book_builder.status.is_valid == False:
            if (datetime.datetime.now() - start_time).total_seconds() > 60:
                print("Book still not initialized after 60 seconds, reconnecting...")
                await self.adapter.reconnect()
                start_time = datetime.datetime.now()
            await asyncio.sleep(1)
        
        start_time = datetime.datetime.now()
        while True:
            snapshot = take_snapshot(self.book_builder, datetime.datetime.now(), product, exchange, num_levels)
            self.buffer.append(snapshot)
            if (datetime.datetime.now() - start_time).total_seconds() > 600: #write to disk every 10 minutes
                write_snapshots(self.buffer, product, exchange, "data")
                self.buffer = []
                start_time = datetime.datetime.now()
            await asyncio.sleep(sample_interval)
            
if __name__ == "__main__":
    collector = Collector()
    asyncio.run(collector.run())
    
