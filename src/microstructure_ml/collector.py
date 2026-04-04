from microstructure_ml.kraken_adapter import KrakenAdapter
from microstructure_ml.book_builder import BookBuilder

class Collector:
    def __init__(self):
        self.adapter = KrakenAdapter()
        self.book_builder = BookBuilder()
    
    async def run(self):
        await self.adapter.connect()
        async for message in self.adapter.listen():
            if message is not None:
                book_updates, update_type = message
                if update_type == "snapshot":
                    self.book_builder.apply_snapshot(book_updates)
                elif update_type == "update":
                    self.book_builder.apply_update(book_updates)
                self.book_builder.health_check()

                if not self.book_builder.is_valid:
                    print("Book is in invalid state, resetting...")
                    self.book_builder.reset()
                    await self.adapter.reconnect()

if __name__ == "__main__":
    import asyncio
    collector = Collector()
    asyncio.run(collector.run())