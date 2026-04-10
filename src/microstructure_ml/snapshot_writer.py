import polars as pl
from datetime import datetime
from pathlib import Path

def write_snapshots(snapshots: list, product: str, exchange: str, base_path: str) -> None:
    df = pl.DataFrame(snapshots)
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    product = product.replace("/","-")
    file_path = Path(base_path) / f"exchange={exchange}" / f"product={product}" / f"date={date_str}" / f"part-{time_str}.parquet"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to: {file_path}")
    print(f"Buffer size: {len(snapshots)}")
    df.write_parquet(file_path)