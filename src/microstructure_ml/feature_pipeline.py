import polars as pl
from pathlib import Path
import argparse

def mid_price() -> pl.Expr:
    return ((pl.col("bid_price_1") + pl.col("ask_price_1")) / 2).alias("mid_price")

def _mid_price_expr() -> pl.Expr:
    return (pl.col("bid_price_1") + pl.col("ask_price_1")) / 2

def spread() -> pl.Expr:
    return (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread")

def _spread_expr() -> pl.Expr:
    return pl.col("ask_price_1") - pl.col("bid_price_1")

def imbalance() -> pl.Expr:
    return ((pl.col("bid_size_1") - pl.col("ask_size_1")) / (pl.col("bid_size_1") + pl.col("ask_size_1"))).alias("imbalance")

def _imbalance_expr() -> pl.Expr:
    return (pl.col("bid_size_1") - pl.col("ask_size_1")) / (pl.col("bid_size_1") + pl.col("ask_size_1"))

def microprice() -> pl.Expr:
    return (_mid_price_expr() + (_imbalance_expr() * _spread_expr() / 2)).alias("microprice")

def depth_imbalance() -> pl.Expr:
    bid_depth = pl.sum_horizontal([pl.col(f"bid_size_{i}") for i in range(1, 11)])
    ask_depth = pl.sum_horizontal([pl.col(f"ask_size_{i}") for i in range(1, 11)])
    return ((bid_depth - ask_depth) / (bid_depth + ask_depth)).fill_nan(None).alias("depth_imbalance")

def rolling_spread(window_size: int) -> pl.Expr:
    return _spread_expr().rolling_mean(window_size, weights=None, min_samples=1).alias(f"rolling_spread_{window_size}")

def compute_features(df: pl.DataFrame, extra_features: list[pl.Expr] | None = None) -> pl.DataFrame:
    if extra_features is None:
        extra_features = []
    default_features = [
        mid_price(),
        spread(),
        imbalance(),
        microprice(),
    ]
    features = default_features + extra_features
    df =  df.with_columns(features)
    return df

def process_file(input_path: Path, input_base: Path, output_base: Path) -> None:
    df = pl.read_parquet(input_path)
    output_path = output_base / input_path.relative_to(input_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = compute_features(df, [depth_imbalance()])
    df.write_parquet(output_path)

def list_data(input_base: Path):
    date_dirs = sorted(
        (p for p in input_base.rglob("date=*") if p.is_dir()), 
        key=lambda p: p.name
        )
    files = []
    for date_dir in date_dirs:
        files.extend(sorted(date_dir.glob("*.parquet")))
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw parquet files to compute features.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input raw parquet file.")
    parser.add_argument("--output", type=Path, required=False, help="Path to the output features parquet file. If not provided, will be saved in the same directory as input with 'features' subdirectory.")
    args = parser.parse_args()
    input_base = args.input
    output_base = args.output if args.output else input_base.parent / "features"
    files = list_data(input_base)
    successes = 0
    for file in files:
        try:
            process_file(file, input_base, output_base)
            successes += 1
        except Exception as e:
            print(f"Error occurred while processing features: {e}")
    print(f"Successfully processed {successes}/{len(files)} files.")
    



    
    
