import polars as pl
from pathlib import Path
import argparse

def compute_labels(df: pl.DataFrame, intervals: list[int]) -> pl.DataFrame:
    columns = [((pl.col("mid_price").shift(-interval) - pl.col("mid_price")) / pl.col("mid_price")).alias(f"return_{interval}s") for interval in intervals]
    df = df.with_columns(columns)
    return df

def process_file(input_path: Path, input_base: Path, output_base: Path, intervals: list[int] | None = None) -> None:
    if intervals is None: 
        intervals = [5, 10, 30]
    df = pl.read_parquet(input_path)
    output_path = output_base / input_path.relative_to(input_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = compute_labels(df, intervals)
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
    parser = argparse.ArgumentParser(description="Process parquet feature files to compute labels")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input features parquet file.")
    parser.add_argument("--output", type=Path, required=False, help="Path to the output labels parquet file. If not provided, will be saved in the same directory as input with 'labels' subdirectory.")
    parser.add_argument('--intervals', nargs='+', type=int, required=False, help = "Time intervals desired for mid price proportional change label calculations")
    args = parser.parse_args()
    input_base = args.input
    output_base = args.output if args.output else input_base.parent / "labels"
    intervals = args.intervals if args.intervals else None
    files = list_data(input_base)
    successes = 0
    for file in files:
        try:
            process_file(file, input_base, output_base, intervals)
            successes += 1
        except Exception as e:
            print(f"Error occured while processing labels: {e}")
    print(f"Successfully processed {successes}/{len(files)} files.")