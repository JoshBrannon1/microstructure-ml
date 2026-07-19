# Microstructure ML — Feature Pipeline

---

## Overview

The feature pipeline is a **batch job** — it reads already-written raw Parquet files, computes microstructure features, and writes enriched Parquet files to `data/features/`. It is a separate module from the collector by design.

**Why separate:**
- **Single responsibility:** the collector collects; the feature pipeline transforms. Mixing them would combine two distinct concerns.
- **Raw data immutability:** raw Parquet is the permanent source of truth. Features are disposable experiments. If a feature formula is wrong or a new feature is needed, raw data can simply be re-fed into a new pipeline. Baking features into the collector would require re-collecting data (days or weeks of waiting) on any formula change.
- **Independent evolution:** the feature pipeline can be modified, re-run, or replaced without touching the collector.

---

## Polars Expression Pattern

All features are implemented as functions returning `pl.Expr` — Polars **lazy expressions**. Expressions are descriptions of transformations; they don't touch data until evaluated against a DataFrame via `df.with_columns()`.

**Why expressions instead of row-by-row functions:**
- Polars executes column operations in optimized Rust — orders of magnitude faster than Python-level loops
- Composable — expressions can be combined to build more complex expressions
- Clean separation between formula definition and data binding

**Aliasing:** `.alias("column_name")` is applied at the end of each public expression to name the output column. Without it, Polars generates a name from the expression string, which is unreadable.

**Private helpers vs. public functions:** Some features (microprice) are composed from simpler features (mid price, spread, imbalance). If you compose an aliased expression inside another, Polars gets confused about column names. Private helpers (`_mid_price_expr`, `_spread_expr`, `_imbalance_expr`) return the raw formula without `.alias()` for use inside compositions. Public functions (`mid_price()`, `spread()`, etc.) add `.alias()` at the outermost level. Convention: underscore prefix signals "internal use only."

**`with_columns()` parallelism:** All expressions passed to a single `df.with_columns()` call are evaluated in parallel against the *original* DataFrame — they cannot see each other's outputs. To reference a newly created column in another expression, a second chained `with_columns()` call is required. This is why `microprice()` uses private `_expr` helpers (inlining mid and imbalance logic) rather than referencing `pl.col("mid_price")`.

---

## Feature Definitions

### Mid Price

**Formula:** `(bid_price_1 + ask_price_1) / 2`

**What it measures:** The geometric midpoint between the best bid and best ask. A *location*, not a distance.

**Common mistake:** `(ask - bid) / 2` measures half the spread (a distance), not the midpoint (a location).

**Why it matters:** Mid price is more stable than either the bid or ask alone. It serves as the reference point for constructing labels — the forward return label compares `mid_price(t + N)` to `mid_price(t)`.

```python
def mid_price() -> pl.Expr:
    return ((pl.col("bid_price_1") + pl.col("ask_price_1")) / 2).alias("mid_price")

def _mid_price_expr() -> pl.Expr:
    return (pl.col("bid_price_1") + pl.col("ask_price_1")) / 2
```

---

### Spread

**Formula:** `ask_price_1 - bid_price_1`

**What it measures:** The distance between best bid and best ask. This is the cost of an immediate round-trip trade (buy at ask, immediately sell at bid).

**Why it matters:** Spread is an indicator of market liquidity. Tight spreads indicate a liquid, competitive market. Wide spreads indicate illiquidity, stress, or uncertainty. Spread widening can precede volatility.

```python
def spread() -> pl.Expr:
    return (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread")

def _spread_expr() -> pl.Expr:
    return pl.col("ask_price_1") - pl.col("bid_price_1")
```

---

### Level 1 Imbalance

**Formula:** `(bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)`

**What it measures:** The relative pressure between resting orders at the top of book. Bounded between -1 and +1.

| Value | Meaning |
|---|---|
| +1 | All pressure on bid side → price likely moving up |
| 0 | Perfectly balanced book |
| -1 | All pressure on ask side → price likely moving down |

**Important distinction:** Imbalance measures *resting* orders (passive limit orders sitting at the best bid/ask), not aggressive orders (market orders that cross the spread and execute immediately).

**Why normalize by `(bid + ask)` rather than use raw difference:**
- Case A: bid_size=10, ask_size=5 → raw difference = 5 (genuine imbalance — one side is double)
- Case B: bid_size=1000, ask_size=995 → raw difference = 5 (nearly perfectly balanced)
- Raw difference is identical; normalization correctly captures *relative* pressure vs. absolute difference

**Why not raw difference for ML:** Features on different scales cause problems for models. A raw size difference could be +847 or -2341 depending on conditions. Models struggle to weight features appropriately when they have wildly different ranges. Normalization produces a consistent, interpretable [-1, +1] scale.

**Research backing:** The imbalance concept is studied by Cont, Kukanov, and Stoikov. Short-horizon price changes are strongly linked to order flow imbalance at the best bid/ask.

```python
def imbalance() -> pl.Expr:
    return (
        (pl.col("bid_size_1") - pl.col("ask_size_1")) /
        (pl.col("bid_size_1") + pl.col("ask_size_1"))
    ).alias("imbalance")

def _imbalance_expr() -> pl.Expr:
    return (pl.col("bid_size_1") - pl.col("ask_size_1")) / (pl.col("bid_size_1") + pl.col("ask_size_1"))
```

---

### Microprice

**Formula:** `mid_price + imbalance * (spread / 2)`

**What it measures:** A weighted mid-price that adjusts toward the side with higher resting volume. It is a better estimate of "fair value" than mid-price alone because it incorporates order book pressure.

**Derivation:**
- Start at mid-price (neutral reference)
- Adjust by some fraction of half the spread
- The fraction is imbalance (bounded [-1, +1])
- Half the spread is the maximum distance from mid to either side

**Extreme cases:**
- Imbalance = +1 (all bid pressure): microprice = mid + spread/2 = ask price
- Imbalance = -1 (all ask pressure): microprice = mid - spread/2 = bid price
- Imbalance = 0 (balanced): microprice = mid price

This behavior is intuitive — when all pressure is on the bid side, the "true" price should be closer to where sellers are willing to transact (the ask).

**Financial interpretation:** Microprice is a high-frequency estimator of the future mid-price, adjusting the current mid for short-term supply/demand imbalance. If microprice > mid price, buyers are more aggressive (larger bid sizes) → upward price pressure expected. This concept is directly used by market makers and quantitative trading firms for fair value estimation.

**Research backing:** Stoikov's microprice materials (used at Gatheral 60) motivate this feature.

**Implementation note:** Uses private `_expr` helpers to avoid Polars alias conflicts when composing expressions.

```python
def microprice() -> pl.Expr:
    return (
        _mid_price_expr() + (_imbalance_expr() * _spread_expr() / 2)
    ).alias("microprice")
```

---

### Depth Imbalance

**Formula:** `(bid_depth - ask_depth) / (bid_depth + ask_depth)`  
where `bid_depth = sum(bid_size_1..10)` and `ask_depth = sum(ask_size_1..10)`

**What it measures:** The same normalized pressure signal as Level 1 imbalance, but across all 10 levels of the book rather than just the top.

**Why it adds information:** Level 1 imbalance only captures the very top of the book. Depth imbalance reveals situations where the top looks balanced but the deeper levels tell a very different story.

**Concrete example:** If Level 1 shows a strong bid imbalance (suggesting upward price movement), but Levels 2-10 show massive ask volume, the upward move may not be sustainable. The deep ask volume will absorb buying pressure before price can move significantly.

**Order book depth and price impact:** More depth means more volume at successive price levels, making it harder for large orders to move price. Less depth means large orders quickly exhaust available liquidity and price must move to find more.

**Polars `sum_horizontal()` vs. `.sum()`:**
- `pl.sum_horizontal()` — sums *across columns* within each row (summing bid_size_1 through bid_size_10 within a single snapshot)
- `.sum()` — aggregates *down rows* within a column
For depth, `sum_horizontal` is correct: we want the total size at all bid levels within one snapshot.

**NaN vs. Null in Polars:** Polars distinguishes between `null` (missing value) and `NaN` (not-a-number, result of invalid arithmetic like 0/0). `sum_horizontal` ignores nulls by default, so a row of all-null sizes produces `0/0 = NaN` rather than `null`. Explicit `.fill_nan(None)` converts NaN to proper Polars nulls for consistent downstream handling.

```python
def depth_imbalance() -> pl.Expr:
    bid_depth = pl.sum_horizontal([pl.col(f"bid_size_{i}") for i in range(1, 11)])
    ask_depth = pl.sum_horizontal([pl.col(f"ask_size_{i}") for i in range(1, 11)])
    return (
        (bid_depth - ask_depth) / (bid_depth + ask_depth)
    ).fill_nan(None).alias("depth_imbalance")
```

---

### Rolling Spread (Implemented, Not Activated)

**Formula:** Rolling mean of `(ask_price_1 - bid_price_1)` over `window_size` rows

**What it measures:** A historical baseline of "normal" spread. Ratio of instantaneous spread to rolling spread can signal market stress, upcoming volatility, or temporary liquidity gaps.

**Window size:** 60 seconds (60 rows at 1-second sampling). Short enough to be responsive, long enough to define a meaningful baseline.

**Boundary problem:** Each file = 10 minutes. Rolling windows spanning file boundaries will have partial windows for the first N rows of each file. Resolved with `min_samples=1` — allows Polars to compute rolling averages on partial windows rather than returning nulls. Tradeoff: slight inaccuracy for the first 60 rows of each file (~10% of rows), deemed acceptable for simplicity.

**Status: Not activated as a default feature.** Rationale: the research question is whether microstructure signal exists at all. Answer this with simpler features first. Rolling spread adds real engineering complexity and its marginal predictive value is unknown until the baseline model runs. Can be passed via `extra_features=[rolling_spread(60)]` when needed.

```python
def rolling_spread(window_size: int) -> pl.Expr:
    return _spread_expr().rolling_mean(window_size, min_samples=1).alias(f"rolling_spread_{window_size}")
```

---

### Why Not Microprice − Mid?

This was considered and explicitly rejected. `microprice − mid` is algebraically equivalent to `imbalance * spread / 2`. It contains no new information beyond what imbalance and spread already encode together. Including it would introduce **multicollinearity** — redundant features can destabilize model weights and make interpretation harder.

---

## `compute_features`

```python
def compute_features(df: pl.DataFrame, extra_features: list[pl.Expr] | None = None) -> pl.DataFrame:
    if extra_features is None:
        extra_features = []
    default_features = [mid_price(), spread(), imbalance(), microprice()]
    features = default_features + extra_features
    return df.with_columns(features)
```

**Default argument as `None` not `[]`:** Python's mutable default argument trap — a default `[]` is created once at function definition time and shared across all calls. If any call mutates it, subsequent calls see the modified list. `None` default with internal `[]` assignment creates a fresh list each call.

**`depth_imbalance` passed via `extra_features`:** It is not in the default list because it requires all 10 size columns to be present. Passing it explicitly keeps the default case simpler.

**All features in one `with_columns()` call:** More efficient than chaining multiple calls. Also ensures all expressions are evaluated against the original DataFrame (not each other's outputs, which would require separate calls).

---

## `process_file`

```python
def process_file(input_path: Path, input_base: Path, output_base: Path) -> None:
    df = pl.read_parquet(input_path)
    output_path = output_base / input_path.relative_to(input_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = compute_features(df, [depth_imbalance()])
    df.write_parquet(output_path)
```

**`Path.relative_to(base)`:** Extracts the portion of a path after the base. Used to mirror partition structure:

```
input_path  = data/raw/exchange=Kraken/product=BTC-USD/date=2026-05-07/part-10-00-00.parquet
input_base  = data/raw
relative    = exchange=Kraken/product=BTC-USD/date=2026-05-07/part-10-00-00.parquet
output_path = data/features/exchange=Kraken/product=BTC-USD/date=2026-05-07/part-10-00-00.parquet
```

This correctly mirrors the full partition structure under any output base — robust to changes in partition depth.

**Evolution:** Started as hardcoded string replace (`"data/raw"` → `"data/features"`). This baked an assumption about where input data lives into a function that shouldn't care about that. `relative_to()` is the correct, portable approach.

---

## `list_data` — File Discovery

```python
def list_data(input_base: Path) -> list[Path]:
    date_dirs = sorted(
        (p for p in input_base.rglob("date=*") if p.is_dir()),
        key=lambda p: p.name
    )
    files = []
    for date_dir in date_dirs:
        files.extend(sorted(date_dir.glob("*.parquet")))
    return files
```

**`rglob("date=*")` approach:** Finds date directories at arbitrary partition depth. If a new partition level is added (e.g., `year=2026/`), the traversal still finds date dirs correctly without code changes.

**Why sort by `p.name` not full path:** Sorts on just the directory name (`date=2025-01-01`) rather than the full path. Correct by intention — works even if partition structure above date dir changes.

**ISO 8601 and lexicographic sort:** Date format `YYYY-MM-DD` was specifically designed so that lexicographic order equals chronological order. Zero-padded fixed-width fields mean string comparison always produces the correct chronological result. No datetime parsing needed.

**Generator syntax gotcha:** When passing a generator expression to `sorted()` alongside a `key=` argument, the generator must be wrapped in its own parentheses:
```python
sorted(
    (p for p in input_base.rglob("date=*") if p.is_dir()),  # ← outer parens required
    key=lambda p: p.name
)
```
Without outer parens, Python misreads the `if` as a ternary expression and raises `SyntaxError: expected 'else' after 'if' expression`.

---

## Entry Point Design

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    
    input_base = args.input
    output_base = args.output or (input_base.parent / "features")
    
    files = list_data(input_base)
    successes = 0
    total = len(files)
    
    for file in files:
        try:
            process_file(file, input_base, output_base)
            successes += 1
        except Exception as e:
            print(f"Error occurred while processing features: {e}")
    
    print(f"Successfully processed {successes}/{total} files.")
```

**Why `argparse` over `sys.argv`:**
- Generates `--help` automatically
- `type=Path` converts string input directly to `Path` objects
- Produces clear error messages when required arguments are missing

**Critical placement:** `parser` and `parse_args()` must live inside `if __name__ == "__main__"`, not at module level. If placed at module level, importing `feature_pipeline` from another module immediately executes `parse_args()`, reads `sys.argv`, finds no `--input`, and errors out.

**`--output` default:** If not provided, defaults to `input_base.parent / "features"`. This means passing `--input data/raw` automatically writes to `data/features` without requiring the user to specify both paths.

**Error handling philosophy:** `try/except` wraps `process_file` *inside* the loop. One corrupted file should not abort processing thousands of others. Catching `Exception` (not specific types) is intentional in a research pipeline where you want to log and continue regardless of what went wrong. Using `exit(1)` inside the except block would defeat the purpose of per-file error handling.

---

## Testing Strategy

Tests were written before implementation (TDD). This forces explicit definition of expected behavior before getting lost in implementation details.

**File:** `tests/test_features_pipeline.py`

- One test per feature function with concrete inputs and hand-verified expected outputs
- Null propagation test: rows with null inputs produce null outputs
- Rolling spread test: verifies partial-window behavior with `min_samples=1`

**Floating point handling:** Polars arithmetic on floats produces values like `-0.19999999999999998` instead of `-0.2` due to binary floating point representation. Tests use `pytest.approx` for numeric comparisons:

```python
result = df.with_columns([mid_price()])
assert result["mid_price"].to_list() == pytest.approx([101.0, 102.0])
```

**Null test design:** `depth_imbalance` needs all 10 bid/ask size columns present even in null tests — otherwise Polars throws a column-not-found error rather than returning null gracefully. Null tests must construct a complete-schema DataFrame even when testing null behavior.

---

## Full Implementation

```python
import polars as pl
from pathlib import Path

def _mid_price_expr() -> pl.Expr:
    return (pl.col("bid_price_1") + pl.col("ask_price_1")) / 2

def _spread_expr() -> pl.Expr:
    return pl.col("ask_price_1") - pl.col("bid_price_1")

def _imbalance_expr() -> pl.Expr:
    return (pl.col("bid_size_1") - pl.col("ask_size_1")) / (pl.col("bid_size_1") + pl.col("ask_size_1"))

def mid_price() -> pl.Expr:
    return _mid_price_expr().alias("mid_price")

def spread() -> pl.Expr:
    return _spread_expr().alias("spread")

def imbalance() -> pl.Expr:
    return _imbalance_expr().alias("imbalance")

def microprice() -> pl.Expr:
    return (_mid_price_expr() + (_imbalance_expr() * _spread_expr() / 2)).alias("microprice")

def depth_imbalance() -> pl.Expr:
    bid_depth = pl.sum_horizontal([pl.col(f"bid_size_{i}") for i in range(1, 11)])
    ask_depth = pl.sum_horizontal([pl.col(f"ask_size_{i}") for i in range(1, 11)])
    return ((bid_depth - ask_depth) / (bid_depth + ask_depth)).fill_nan(None).alias("depth_imbalance")

def rolling_spread(window_size: int) -> pl.Expr:
    return _spread_expr().rolling_mean(window_size, min_samples=1).alias(f"rolling_spread_{window_size}")

def compute_features(df: pl.DataFrame, extra_features: list[pl.Expr] | None = None) -> pl.DataFrame:
    if extra_features is None:
        extra_features = []
    default_features = [mid_price(), spread(), imbalance(), microprice()]
    return df.with_columns(default_features + extra_features)

def process_file(input_path: Path, input_base: Path, output_base: Path) -> None:
    df = pl.read_parquet(input_path)
    output_path = output_base / input_path.relative_to(input_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = compute_features(df, [depth_imbalance()])
    df.write_parquet(output_path)
```

---

## Pipeline Run Results

```
make features INPUT=data/raw
Successfully processed 3335/3335 files.   ← original run

make features INPUT=data/raw
Successfully processed 2319/2326 files.   ← after data migration (7 corrupt files)
```

The 7 corrupt files were Parquet files being written when the collector was interrupted mid-write. These files are discarded by `process_file`'s error handling and do not appear in downstream data. Less than 0.3% loss — acceptable for research purposes.
