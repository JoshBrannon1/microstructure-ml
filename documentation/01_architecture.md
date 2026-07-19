# Microstructure ML — Architecture & System Design

---

## Research Question

**Original:** How much short-horizon predictive signal exists in top-of-book microstructure features for BTC-USD, and how stable is it out-of-sample under leakage-aware validation?

**Revised (evidence-driven reframe, see `05_findings.md`):** How well can top-of-book microstructure features predict microprice deviation from mid price at short horizons for BTC-USD, and how stable is that signal out-of-sample under leakage-aware walk-forward validation?

---

## Data Flow

```
Kraken WebSocket
      ↓
KrakenAdapter       ← parses raw WebSocket messages into BookUpdate NamedTuples
      ↓
BookBuilder         ← maintains live L2 order book state; enforces invariants
      ↓
Collector           ← wires components; manages sampling loop and resyncs
      ↓
SnapshotSampler     ← extracts top-10 bid/ask levels into a flat dict each second
      ↓
SnapshotWriter      ← buffers snapshots; writes partitioned Parquet every 10 minutes
      ↓
[data/raw/]         ← raw Parquet files, partitioned by exchange/product/date
      ↓
FeaturePipeline     ← batch job; computes microstructure features; writes features/
      ↓
[data/features/]
      ↓
LabelBuilder        ← batch job; computes forward return labels; writes labels/
      ↓
[data/labels/]      ← full dataset: raw + features + labels in one Parquet per file
      ↓
ValidationSplits    ← generates leakage-aware walk-forward split definitions
      ↓
TrainingPipeline    ← loads splits, trains model, evaluates, reports
```

---

## Module Summary

| Module | File | Responsibility |
|--------|------|----------------|
| KrakenAdapter | `kraken_adapter.py` | WebSocket connection, message parsing, normalization |
| CoinbaseAdapter | `coinbase_adapter.py` | Deprecated; defines `BookUpdate` NamedTuple used everywhere |
| BookBuilder | `book_builder.py` | Maintains live L2 order book state; health checks; resync |
| Collector | `collector.py` | Wires all components; manages sampling loop and resync logic |
| SnapshotSampler | `snapshot_sampler.py` | Extracts top-N book levels into flat dict |
| SnapshotWriter | `snapshot_writer.py` | Writes buffered snapshots to partitioned Parquet |
| FeaturePipeline | `feature_pipeline.py` | Batch: raw → features Parquet |
| LabelBuilder | `label_builder.py` | Batch: features → labels Parquet |
| ValidationSplits | `validation_splits.py` | Walk-forward split definitions |
| TrainingPipeline | `training_pipeline.py` | Load, train, evaluate, report |

---

## File Structure

```
~/microstructure-ml/
├── .venv/                             ← Poetry-managed virtualenv
├── src/
│   └── microstructure_ml/
│       ├── __init__.py
│       ├── coinbase_adapter.py        ← BookUpdate NamedTuple + deprecated adapter
│       ├── kraken_adapter.py          ← Active Kraken WebSocket adapter
│       ├── book_builder.py            ← BookStatus + BookBuilder
│       ├── collector.py               ← Main entry point, async event loop
│       ├── snapshot_sampler.py        ← take_snapshot()
│       ├── snapshot_writer.py         ← write_snapshots()
│       ├── feature_pipeline.py        ← Feature expressions + compute_features()
│       ├── label_builder.py           ← compute_labels() + process_file()
│       ├── validation_splits.py       ← create_splits() + train_val_splits()
│       └── training_pipeline.py       ← load_split(), train_model(), evaluate_model()
├── tests/
│   ├── test_book_builder.py
│   ├── test_snapshot_sampler.py
│   ├── test_snapshot_writer.py
│   ├── test_features_pipeline.py
│   ├── test_label_builder.py
│   ├── test_validation_splits.py
│   └── test_training_pipeline.py
├── configs/                           ← Future run parameters
├── data/                              ← Gitignored; all Parquet data here
│   ├── archive/raw/                   ← Preserved older/lower-quality raw data
│   ├── raw/                           ← Current raw snapshots from Kraken
│   ├── features/                      ← Raw + engineered features
│   └── labels/                        ← Raw + features + labels (full dataset)
├── reports/                           ← Future plots and writeup
├── documentation/                     ← Project documentation (this directory)
├── Makefile
├── README.md
├── DESIGN.md
├── DATA.md
├── pyproject.toml
└── poetry.lock
```

---

## Data Layout

All three data tiers use identical Hive partition structure:

```
data/<tier>/exchange=Kraken/product=BTC-USD/date=YYYY-MM-DD/part-HH-MM-SS.parquet
```

**Hive partitioning** encodes key=value pairs in folder names. Columnar query engines (Polars, Spark, DuckDB) can parse these folder names as column values and skip entire directories when filtering by date — this is called **predicate pushdown**.

The labels tier is the full dataset — raw columns, feature columns, and label columns are all present in the same Parquet file. This avoids joins at training time.

---

## Data Schema

### Raw Parquet (43 columns)

| Column group | Columns | Type |
|---|---|---|
| Bid prices | `bid_price_1` … `bid_price_10` | Float64 |
| Bid sizes | `bid_size_1` … `bid_size_10` | Float64 |
| Ask prices | `ask_price_1` … `ask_price_10` | Float64 |
| Ask sizes | `ask_size_1` … `ask_size_10` | Float64 |
| Metadata | `timestamp`, `product`, `exchange` | Datetime[μs], String, String |

- Bids sorted high → low (best bid = `bid_price_1`)
- Asks sorted low → high (best ask = `ask_price_1`)
- Invariant: `bid_price_1 < ask_price_1` on every valid row

### Features Added (5 columns)

`mid_price`, `spread`, `imbalance`, `microprice`, `depth_imbalance`

### Labels Added (3 columns)

`return_5s`, `return_10s`, `return_30s`

Total in labels Parquet: **51 columns**

---

## Component Design Details

### BookUpdate NamedTuple

Defined in `coinbase_adapter.py`, imported across the codebase. Provides a normalized, immutable representation of any order book update regardless of source exchange.

```python
class BookUpdate(NamedTuple):
    side: str           # "bid" or "ask"
    price: float        # price level
    size: float         # quantity at level; 0 means remove level
    time: Optional[str] # timestamp from exchange
```

**Why NamedTuple:** Immutable (prevents accidental mutation), lightweight, provides named field access. No behavior needed — just structured data.

**Adapter abstraction value:** `BookUpdate` provides a normalized interface between any exchange adapter and `BookBuilder`. Adding a new exchange only requires writing a new adapter that returns `(list[BookUpdate], message_type)` tuples. Everything downstream is unchanged.

---

### BookStatus NamedTuple

Defined in `book_builder.py`.

```python
class BookStatus(NamedTuple):
    is_valid: bool
    reason: Optional[str]
    is_crossed: bool
    is_anomalous_spread: bool
```

**Evolution:** Started as `self.is_valid: bool`. Upgraded to `BookStatus` after identifying that two different invalidation sources (crossed book, anomalous spread) were indistinguishable from a boolean alone. The `reason` field lets the caller log *why* the book became invalid before resetting.

**Crossed book handling (key fix):** Originally, `best_bid >= best_ask` triggered a full resync. This caused ~80 null rows per 10-minute file from the resync window. Investigation showed these crossings were small ($2.80) and are timing/ordering artifacts during high-speed trading — real market signal, not corruption. Fix: crossed book now sets `is_valid=True, is_crossed=True` (flagged but not invalidated).

Three-case logic in `update_best_prices()`:
1. Either side is `None` → `is_valid=False`, reason="Empty side" — a genuinely invalid state
2. `best_bid >= best_ask` → `is_valid=True, is_crossed=True` — flagged but data continues flowing
3. `best_bid < best_ask` → `is_valid=True, is_crossed=False` — normal state

**NamedTuple immutability note:** Status updates require full reassignment: `self.status = BookStatus(False, "reason", False, False)`.

---

### BookBuilder

Maintains `bids: dict[float, float]` and `asks: dict[float, float]` where keys are prices and values are sizes.

**Key methods:**

- `apply_snapshot(updates)` — clears book entirely, rebuilds from snapshot, seeds spread history, sets valid. Called on initial connection and after each resync.
- `apply_update(updates)` — returns early if book is invalid; applies changes; removes size=0 levels (Kraken's way of deleting a price level); calls `update_best_prices()`
- `update_best_prices()` — recomputes `best_bid = max(bids)` and `best_ask = min(asks)`; applies three-case crossed-book logic above
- `reset()` — clears all state, clears spread history, sets `BookStatus(False, "Book reset, waiting for next snapshot", False, False)`

**Initialization:** `BookStatus(False, "Book not initialized, waiting for first snapshot", False, False)` — an empty book is explicitly invalid because `take_snapshot` would produce a dict full of `None` values.

**Health check evolution:**
- Originally: flagged anomalous spreads (>2x rolling average of last 10) as invalid → triggered constant resyncs on BTC which has naturally wide and volatile spreads
- Decision: removed spread anomaly detection entirely pending real data collection
- Rationale: unusual spread is a real market event worth recording, not a corruption indicator. Hard invariants (crossed book) are still enforced because a crossed book is a sequencing artifact.

---

### KrakenAdapter

- Connects to `wss://ws.kraken.com/v2`
- Subscribes to `book` channel for `BTC/USD`
- `parse_message()` returns `(list[BookUpdate], message_type)` or `None`
- `listen()` is an async generator yielding `(updates, message_type)` tuples
- `reconnect()` closes existing connection and calls `connect()` again
- Non-book messages filtered via `message.get("channel") != "book"`

**Why Kraken over Coinbase:** Coinbase Level 2 requires authentication. Kraken's public WebSocket feed requires only a subscription message. The adapter abstraction means switching exchanges only requires writing a new adapter.

---

### Collector

Two async methods managing the event loop:

**`run()`:**
- Creates WebSocket connection via adapter
- Launches `sample_loop` as concurrent task via `asyncio.create_task()`
- Processes incoming messages from `adapter.listen()`, routes to `BookBuilder`
- If book becomes invalid: logs reason → resets book → reconnects adapter

**`sample_loop()`:**
- Waits for book to become valid with 60-second timeout before reconnecting
- Checks `self.book_builder.status.is_valid` before each snapshot (null guard — prevents writing null rows during resync)
- Takes snapshot every `sample_interval` seconds (1 second)
- Appends to buffer
- Writes buffer to Parquet every `buffer_time` seconds (600 seconds = 10 minutes) using flush timer pattern

**Flush timer pattern:**
```python
start_time = datetime.datetime.now()
# inside loop:
if (datetime.datetime.now() - start_time).total_seconds() > 600:
    write_snapshots(self.buffer, ...)
    self.buffer = []
    start_time = datetime.datetime.now()
await asyncio.sleep(sample_interval)
```

**Why write-then-sleep, not sleep-then-write:** If the program crashes during sleep, the write has already happened. Reversing the order risks losing a full buffer on crash.

**Product normalization:** `self.product = self.adapter.symbol.replace("/", "-")` — Kraken uses `BTC/USD` internally; dataset and folder names use `BTC-USD`. Normalization happens at the `Collector` level.

---

### SnapshotSampler

`take_snapshot(book, timestamp, product, exchange, num_levels) -> dict`

- Sorts bids descending, asks ascending
- Slices top `num_levels` from each side
- Builds flat dict: `bid_price_1..N`, `bid_size_1..N`, `ask_price_1..N`, `ask_size_1..N`
- Pads missing levels with `None` (defensive — guarantees consistent schema regardless of book depth)
- Appends `timestamp`, `product`, `exchange`
- Total keys: `N * 4 + 3` (43 for N=10)

**Why flat dict:** Each column represents a consistent semantic value across all rows (e.g., `bid_price_1` is always the best bid). Enables vectorized operations in Polars. Could not do `df["bid_price_1"].mean()` efficiently with nested structures.

---

### SnapshotWriter

`write_snapshots(snapshots, product, exchange, base_path) -> None`

- Converts list of flat dicts to `pl.DataFrame`
- Partition path: `data/raw/exchange=<exchange>/product=<product>/date=YYYY-MM-DD/part-HH-MM-SS.parquet`
- Creates directories with `mkdir(parents=True, exist_ok=True)`
- Files named by write time; each covers exactly 10 minutes of data (~600 rows)

---

## Key Design Decisions

### WebSocket + Asyncio (Push vs. Poll)

**Push-based streaming** via WebSocket means Kraken sends updates the moment anything changes. BTC/USD can have hundreds of book updates per second; polling at fixed intervals would miss the vast majority.

**Asyncio** provides single-threaded cooperative concurrency — not multithreading. The program switches between tasks at `await` points. When one task waits (e.g., for the next WebSocket message), another task runs. This allows message processing, 1-second sampling, and 10-minute writes to run concurrently on one thread.

**Critical distinction:** Asyncio ≠ multithreading. Threads run truly in parallel on different CPU cores. Asyncio switches between tasks on one thread at `await` points. This is correct here because the workload is I/O-bound (waiting for WebSocket messages), not CPU-bound.

### Parquet over CSV

| Feature | Parquet | CSV |
|---|---|---|
| Storage format | Columnar | Row-based |
| I/O for column ops | Read only needed columns | Must read all bytes |
| Type information | Stored with data | None (all strings) |
| Compression | Highly efficient | None |
| Predicate pushdown | Yes (with partitioning) | No |
| Human readable | No | Yes |

**Predicate pushdown:** The `date=YYYY-MM-DD` partition folder structure lets query engines skip entire directories without opening files. Querying one day from 180 days of data only reads 1/180th of files.

**Small files problem:** Writing one file per snapshot (86,400/day) creates massive filesystem overhead. Writing every 10 minutes produces 144 files/day — manageable. The 600x reduction in file count significantly reduces per-file metadata overhead on reads.

### Hard Invariants vs. Soft Signals

**Hard invariant — crossed book (`best_bid >= best_ask`):**
- Physically impossible in a real market (matching engine would execute the trade immediately)
- If seen in local state: indicates a sequencing/timing artifact, not market reality
- Original response: reset and reconnect (caused too many resyncs, too many null rows)
- Updated response: flag with `is_crossed=True` but continue collecting data

**Soft signal — unusual spread:**
- Real market events (volatile periods, thin liquidity, flash events)
- Worth recording in the dataset — potentially predictive features
- Response: do NOT invalidate; let the data flow
- Health check threshold removed entirely pending data distribution analysis

### Batch vs. Daemon Architecture

The collector is a **daemon** — a long-running process that continuously receives data and periodically flushes to disk. The feature pipeline and label builder are **batch jobs** — they read already-existing files, transform them, and exit. This is a fundamentally different execution model, and the design reflects that. Mixing feature computation into the collector would violate single responsibility and make it impossible to reprocess features without re-collecting data.

### Raw Data Immutability

Raw Parquet files are the permanent source of truth. Features are disposable experiments. If a feature formula is wrong or a new feature is needed, raw data can simply be re-fed into a new pipeline. Features being separate means reprocessing is just re-running a script — no re-collection (days or weeks of waiting).

---

## Known Limitations & Deferred Work

| Limitation | Impact | Plan |
|---|---|---|
| No exponential backoff on reconnection | Could contribute to thundering herd on exchange outages | Implement in hardening phase |
| `print()` instead of `logging` | No timestamps, log levels, or file output in production | Replace with `logging` module |
| Single exchange and product | Can't collect multi-exchange or multi-product data | Refactor `Collector` to manage dict of `(adapter, book_builder, buffer)` per product |
| `is_crossed` not stored as column in snapshots | Can't use as feature; can't backfill existing files easily | Add to `take_snapshot()`, backfill if model training reveals it matters |
| Health check spread threshold removed | Can't flag unusual spread events in real-time | Revisit after analyzing collected data distribution |
| No reconnection attempt limit | Could retry indefinitely | Add max retries with exponential backoff |
| Rolling spread window bridges files | First 60 rows of each file have partial-window rolling spread | Rework if rolling spread proves predictively useful |
| `asyncio` deprecation warning from `websockets.legacy` | Noise in logs | Suppress or migrate to newer API |

---

## Bugs Fixed During Development

| Bug | Root Cause | Fix |
|---|---|---|
| Buffer never cleared after write | `self.buffer = []` missing after write | Added after `write_snapshots()` call |
| Write timing used counter `i % 600` | Exact match on loop counter unlikely | Replaced with flush timer using `timedelta` comparison |
| Write occurred after sleep | Wrong order of operations | Moved write before `asyncio.sleep()` |
| `sample_loop` returned immediately | Book always invalid at startup by design | Replaced early return with `while not valid: await sleep(1)` wait loop |
| Spread history inflated by anomalous spreads | Anomalous spreads were added to history, raising the average | Added check before append (later removed with health check) |
| `total_seconds` called without parentheses | Property vs. method confusion | Added `()` — `timedelta.total_seconds()` is a method |
| `start_time` not reset after flush | Next write would happen immediately | Reset `start_time = datetime.datetime.now()` after write |
| Product string inconsistency | Kraken uses `BTC/USD`, dataset needs `BTC-USD` | Normalized at `Collector.__init__` |
| `__init__` status set to `is_valid=True` | Empty book has no data | Changed to `BookStatus(False, "Book not initialized...")` |
| Null rows from crossed-book resyncs | Crossed book triggered resync, null window during recovery | Changed crossed book to `is_valid=True, is_crossed=True`; added null guard in `sample_loop` |
| Hardcoded absolute path in `collector.py` | `/home/josh/...` doesn't exist on GCP VM | Changed to relative path `"data"` |
| Duplicated path after `gcloud scp` | `scp` nests source inside destination when destination exists | Moved contents up one level with `mv` |
