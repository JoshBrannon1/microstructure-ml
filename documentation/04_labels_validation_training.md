# Microstructure ML — Labels, Validation & Training Pipeline

---

## Part 1: Label Builder

### What a Label Is

A label is a concrete value attached to each row of the dataset *before* any model sees it. It is computed from historical data after the fact and represents the ground truth outcome the model is trained to predict. The model learns to map features → label. The label is not a prediction — it is what actually happened.

**Key distinction on leakage:** During dataset construction you have access to all historical data, including the future. The leakage risk is not in the label itself — it is in whether future information sneaks into the *features*. Features must only use information available at time `t`. Labels are ground truth computed after the fact; they are supposed to use `t+N`.

### Forward Return Formula

```
return_N(t) = (mid_price(t+N) - mid_price(t)) / mid_price(t)
```

**Why divide by `mid_price(t)`:**
Normalizes the change relative to where price currently sits — the quantity we want to predict against. Dividing by `mid_price(t+N)` would give change relative to the new value, which is less intuitive. Proportional return is more meaningful than raw price difference because BTC at ~$100k: a $10 move means something very different at different price levels.

**Why not divide by N:**
Dividing by time gives a rate of change per second, making labels incomparable across different horizon lengths. The label is simply: "how much did price move in N seconds from this moment?"

### Multiple Horizons

**Decision: Compute labels for multiple horizons in one pass.**

Intervals: `return_5s`, `return_10s`, `return_30s` (20s optionally via `--intervals` flag)

**Rationale:** The research question implies comparing horizons. Does a 5-second horizon have more signal than a 30-second horizon? Computing all at once costs nothing extra and enables coarse-to-fine tuning. Finding the best rough horizon first, then fine-tuning within a narrower range.

**Polars `shift(-N)` for vectorized label computation:**
```python
columns = [
    ((pl.col("mid_price").shift(-interval) - pl.col("mid_price")) / pl.col("mid_price"))
    .alias(f"return_{interval}s")
    for interval in intervals
]
df = df.with_columns(columns)
```

`shift(-N)` moves column values upward by N positions, filling the tail with nulls. This computes all labels in a single vectorized expression — dramatically faster than row-by-row iteration.

### Edge Handling

**Decision: Leave nulls in place at construction time; drop at training time.**

The last N rows of every file cannot have a label for `return_Ns` because `t+N` does not exist within that file. Polars `shift(-N)` automatically places nulls in those positions.

**Why not bridge across files:** Files are 10-minute blocks. Bridging would require reading the next file to get rows for the current file's tail — complex and not worth it. With 600 rows per file and N=30, losing 30 rows is 5% of that file. Across 2319 files that is ~70,000 rows lost, but total dataset is ~1.4 million rows. Acceptable.

**Why not impute:** Imputing a fake label would introduce its own bias. Nulls are left in place and dropped at training time with `df.drop_nulls()`.

**Null rows expected at tail:**
```
shape: (5, 51)
│ depth_imbalance ┆ return_5s ┆ return_10s ┆ return_30s │
│ -0.533371       ┆ null      ┆ null       ┆ null       │
│ -0.533371       ┆ null      ┆ null       ┆ null       │
```

### Regression vs. Classification

**Regression chosen as primary label type.** Classification remains viable and is used in much academic microstructure literature.

| Approach | Pros | Cons |
|---|---|---|
| Regression | Continuous signal, no discretization needed, answers "how much" | Noisy at 1-second resolution, harder to evaluate |
| Classification (up/down/flat) | Coarser question easier to answer reliably, robust to noise | Requires threshold decision, class imbalance risk |

The research question asks "how much signal exists" which implies regression. However, the classification threshold decision for the "flat" bucket (returns smaller than epsilon ε) has real consequences for dataset balance and what the model learns.

### Module: `label_builder.py`

**`compute_labels(df, intervals) -> pl.DataFrame`**
- Builds list of Polars expressions via list comprehension, one per interval
- Single `df.with_columns(columns)` call — all labels computed in one pass
- Does not accept `None` for intervals; defaults live in `process_file`, not here

**`process_file(input_path, input_base, output_base, intervals=None) -> None`**
- Default intervals `[5, 10, 30]` live here
- Mirrors partition structure using `input_path.relative_to(input_base)` (same pattern as feature pipeline)
- Accepts `intervals` as parameter for CLI configurability

**`list_data(input_base) -> list[Path]`**
- Same pattern as feature pipeline: `rglob("date=*")`, sorted by name, files sorted within each date

**Entry point:**
```
make labels INPUT=data/features [INTERVALS="5 10 30"]
Successfully processed 2319/2319 files.
```

---

## Part 2: Book Builder — Crossed Book Fix

### Problem Discovered During Label Inspection

After running `make labels`, inspection revealed ~80 null rows per file in `mid_price` and all label columns. Investigation showed:
- Null clusters were consecutive seconds, not random
- `bid_price_1` and `ask_price_1` were both null simultaneously
- 7 distinct null clusters per 10-minute file
- Nulls coincided exactly with book resync events

### Root Cause

`update_best_prices()` marked the book invalid when `best_bid >= best_ask` (crossed book), triggering a full resync. Each resync takes several seconds waiting for a new Kraken snapshot, during which `best_bid` and `best_ask` are both `None`. The snapshot sampler was writing null rows during this window.

Actual crossing values were small: `best_bid=81331.7 >= best_ask=81328.9` (~$2.80 cross). These are timing/ordering artifacts during high-speed trading — Kraken briefly reports a crossed state during rapid updates, not a real data integrity issue.

### Key Insight

A crossed book during high volatility is potentially *signal*, not noise. These may occur during interesting market microstructure events (large trades crossing the book, rapid quote updates). Discarding the data and reconnecting was actively harming the dataset.

### Fix: `is_crossed` Field in BookStatus

```python
class BookStatus(NamedTuple):
    is_valid: bool
    reason: Optional[str]
    is_crossed: bool
    is_anomalous_spread: bool
```

`update_best_prices()` now has three cases:
1. Either side is `None` → `is_valid=False` (genuinely no book — nothing to snapshot)
2. `best_bid >= best_ask` → `is_valid=True, is_crossed=True` (flagged but data flows)
3. `best_bid < best_ask` → `is_valid=True, is_crossed=False` (normal state)

### Fix: Null Guard in `collector.py`

Added validity check before each snapshot:

```python
if self.book_builder.status.is_valid:
    snapshot = take_snapshot(...)
    self.buffer.append(snapshot)
```

Previously, snapshots were taken regardless of book validity, writing null rows during the brief resync window (before the crossed-book fix, this was happening frequently).

### Deferred Work

- Adding `is_crossed` as a feature column in `take_snapshot()` and `snapshot_sampler.py`
- Backfilling existing raw Parquet files with `is_crossed=False`
- Decision: defer until model training reveals whether crossed book data matters for prediction

---

## Part 3: Validation Framework

### Core Concepts

#### Why Train/Test Splits Exist

A model trained on all available data and evaluated on the same data will overfit — it memorizes rather than generalizes. The test set simulates truly unseen data to measure whether the model learned real patterns.

#### Two Distinct Failure Modes

| Problem | Cause | Effect |
|---|---|---|
| Overfitting | Model memorizes training data | Fails to generalize to new data |
| Lookahead leakage | Model trained on future information | Learned a pattern that cannot exist in live trading |

A leaked model is not just overfit — it is fundamentally broken. It never actually learned the prediction task. No amount of regularization can fix a leaked model.

#### Why Standard K-Fold Fails for Time Series

Standard k-fold randomly assigns rows to folds. With a 5-day dataset, Fold 2 might: train on days 1, 3, 4, 5 — validate on day 2. This trains on day 5 (future) to evaluate day 2 (past). The model learns patterns from data that would not exist at prediction time. Validation metrics are inflated and meaningless.

**scikit-learn explicitly states:** `TimeSeriesSplit` exists because other CV methods can cause training on future data and evaluating on past data.

#### Train / Validation / Test — Three-Way Split

| Split | Purpose | How often examined |
|---|---|---|
| Training | Fits model weights/parameters | Every iteration |
| Validation | Guides hyperparameter and model selection decisions | Many times |
| Test | Final unbiased performance estimate | Exactly once, at the very end |

Every time you look at test performance and make a decision based on it, you contaminate the test set — it becomes another form of training data. Validation exists so you can iterate without touching the test set.

### Walk-Forward Validation

**Rule: The training set always ends before the validation set begins. The boundary only moves forward.**

Structure with expanding window:

```
Split 1:  train=[day1]                → val=[day2, day3, day4]
Split 2:  train=[day1, day2]          → val=[day3, day4, day5]
Split 3:  train=[day1, day2, day3]    → val=[day4, day5, day6]
...
```

**Expanding vs. Fixed (Rolling) Window:**

| Approach | Pros | Cons |
|---|---|---|
| Expanding | More data for later splits, simulates real deployment | Computationally heavier; older data may hurt if market regime changed |
| Fixed (rolling) | Adapts to regime changes, consistent training set size | Discards historical data, smaller training sets |

**Decision: expanding window.** If per-split validation performance degrades in later splits, investigate fixed windows as a response to potential regime change.

### Test Set Design

**Decision: Hold out the most recent N days as the test set.**

Rationale: simulates real deployment — train on everything known up to now, predict on what comes next. Taking test data from the middle of the dataset reintroduces leakage (walk-forward training would train on data after the test period).

### Label Overlap / Gap Concern

Forward labels create potential overlap at split boundaries. `return_30s` for a row at time `t` uses data from `t+30`. If training ends at second `t` and validation starts at `t+1`, row `t` in training overlaps with the first 29 rows of validation.

**Resolution:** With day-level splits, the boundary between training and validation is 24 hours — far larger than the 30-second maximum label horizon. Overlap concern is automatically satisfied at day-level granularity. No explicit purge/embargo needed. (This would need revisiting for sub-day splits.)

### Split Parameters (Current Defaults)

```
VAL_DAYS  = 3    # days in each validation window
STEP_DAYS = 1    # how far validation window advances between splits
TEST_DAYS = 5    # days reserved for final held-out test set
```

**With 19 days of data:**
- 5 days held out as test → 14 days for train/val
- `range(1, 14 - 3 + 1, 1) = range(1, 12, 1)` → 11 walk-forward splits

**Why `step_days=1`:** Maximizes number of splits for a small dataset. More splits = more resolution on stability over time. Computational cost is negligible for this dataset size.

**Why `val_days=3`:** A single day of validation (`val_days=1`) could be skewed by one unusual day. Three days gives a more stable MSE estimate per split.

**Why `test_days=5`:** Preserves enough held-out data for a meaningful final evaluation, including the post-gap period (May 27-29). The 4-day gap (May 23-26) was retained intentionally — testing across a potential distribution shift is a stronger out-of-sample evaluation.

### Module: `validation_splits.py`

**`list_dates(input_base: Path) -> list[Path]`**
- Returns sorted list of `date=*` directories
- Same pattern as `list_data` but returns directories, not individual files

**`train_val_splits(dates, val_days, step_days) -> list`**
- Pure logic function — no filesystem access, fully unit-testable with plain lists
- `range(1, len(dates) - val_days + 1, step_days)` generates split start indices
- Upper bound ensures the full validation window always fits within available dates
- Returns list of `(train_dates, val_dates)` tuples

```python
def train_val_splits(dates, val_days, step_days):
    splits = []
    for i in range(1, len(dates) - val_days + 1, step_days):
        splits.append((dates[:i], dates[i:i+val_days]))
    return splits
```

**`create_splits(input_base, val_days, step_days, test_days) -> tuple[list, list]`**
- Validates: `test_days + val_days + 1 <= len(dates)`, raises `ValueError` otherwise
- Reserves `dates[-test_days:]` as held-out test set
- Calls `train_val_splits` on remaining dates
- Returns `(walk_forward_splits, test_dates)`

**Separation of concerns:** `validation_splits.py` returns split *definitions* only — lists of date paths. It does not load data. Loading is the training loop's responsibility. This keeps the validation framework reusable regardless of model choice.

---

## Part 4: Training Pipeline

### Feature Selection (X columns)

Only computed features are fed to the model — raw price and size columns are excluded:

| Column | Include | Reason |
|---|---|---|
| `mid_price` | yes | Core microstructure feature |
| `spread` | yes | Core microstructure feature |
| `imbalance` | yes | Core microstructure feature |
| `microprice` | yes | Core microstructure feature |
| `depth_imbalance` | yes | Core microstructure feature |
| `bid_price_1..10` | no | Non-stationary; absolute price is not a microstructure signal |
| `ask_price_1..10` | no | Non-stationary |
| `bid_size_1..10` | no | Redundant with `depth_imbalance` and `imbalance` |
| `ask_size_1..10` | no | Redundant |
| `timestamp`, `product`, `exchange` | no | Metadata, not predictive features |

**Why exclude raw prices:** BTC price is non-stationary. A model trained when BTC was at $30k would see $81k as an extreme outlier. Microstructure features like imbalance, spread, and microprice are normalized ratios that capture supply/demand dynamics independent of absolute price level — they generalize across price regimes.

### Why Linear Regression as Baseline

1. **Interpretability:** Each coefficient directly shows the direction and magnitude of each feature's effect on the predicted return. For 5 features, you get 5 numbers that tell a complete story.
2. **Occam's Razor (establish the simpler explanation first):** If linear regression already captures most of the signal, gradient boosted trees are unnecessary complexity. If trees significantly outperform, *that* is the evidence of nonlinear structure worth pursuing.
3. **Research validity:** Skipping to complex models means you don't know if a simpler linear combination would suffice. Baselines are how research is supposed to start.

**Gradient boosted trees by comparison:** Feature importance tells you *how much* a feature was used in splits, but not the direction or magnitude of its effect on predictions. SHAP values get closer but are still more indirect. The weakness is **directional interpretability**, not magnitude.

### `MODELS` Dictionary

```python
MODELS = {
    "linear": LinearRegression,
    "rf": RandomForestRegressor,
}
```

**Why a dict instead of dynamic string-to-class loading:** Simpler, fails loudly with a clear `KeyError` on unknown model names, gives an explicit inventory of supported models. Avoids the complexity of `importlib` or `getattr` for what is realistically 2-3 models.

### `load_split`

```python
def load_split(data: list[Path], keep_columns: list[str], label: str) -> tuple[NDArray, NDArray]:
    files = []
    frames = []
    for date_dir in data:
        files.extend(sorted(date_dir.glob("*.parquet")))
    for file in files:
        frames.append(pl.read_parquet(file).select(keep_columns + [label]))
    if not frames:
        raise ValueError(f"No parquet files found in {data}")
    df = pl.concat(frames)
    df = df.drop_nulls()
    if df.is_empty():
        raise ValueError(f"No data remaining after dropping nulls for dates: {data}")
    return df.drop(label).to_numpy(), df.get_column(label).to_numpy()
```

**Key decisions:**
- **Select columns before concat:** Narrower frames = cheaper concat (less data moved in memory)
- **Single `pl.concat()` after loop:** N in-loop concats cause N memory allocations; one post-loop concat is O(1) allocations
- **Drop nulls after concat:** One bulk operation instead of N small ones
- **`keep_columns` as parameter:** Caller decides which features to use. Modularity without needing to know column names inside the function.
- **Single `label` parameter:** Each experiment targets one horizon. Multiple labels would train a model for two different things simultaneously.
- **Defensive guards:** `ValueError` with clear message if no files found, or if all rows are null

**`y = df.get_column(label).to_numpy()`** — `get_column` returns a Polars Series, `.to_numpy()` converts it. Returns a 1D array (correct for sklearn regression).

**`X = df.drop(label).to_numpy()`** — drops the label column, converts the remaining feature columns to a 2D numpy array (n_samples × n_features).

### `train_model`

```python
def train_model(x_train: NDArray, y_train: NDArray, model_class: type = LinearRegression, **kwargs) -> BaseEstimator:
    model = model_class(**kwargs)
    model.fit(x_train, y_train)
    return model
```

**Sklearn pattern:** All sklearn models follow: instantiate → `fit()` → `predict()`. `fit()` handles the entire training loop internally (gradient descent, weight updates, convergence). Architecture is configured at instantiation via hyperparameters, not modified during `fit()`.

**`**kwargs`:** Passes arbitrary named arguments to the model constructor. A caller running `RandomForestRegressor` can pass `n_estimators=100, max_depth=5` without any changes to `train_model`. Different models need different hyperparameters — `**kwargs` handles all cases without knowing each model's API upfront.

**`LinearRegression` default:** Chosen as the baseline starting point. Having a default makes sense for a research project where you're the only caller.

### `evaluate_model`

```python
def evaluate_model(model: BaseEstimator, x_val: NDArray, y_val: NDArray) -> float:
    return mean_squared_error(y_val, model.predict(x_val))
```

**Why MSE:**
- Standard for regression tasks
- Penalizes large errors more than small ones (squaring) — appropriate because large prediction errors in a trading context are disproportionately costly
- Sensitive to outliers (BTC has flash crashes, liquidation cascades) — this is a known limitation

**Alternatives considered:**
- MAE: treats all error magnitudes equally; doesn't capture the asymmetric cost of large errors
- MSLE: requires non-negative values; forward returns can be negative → not viable
- Directional accuracy: ignores magnitude entirely; loses information about how wrong predictions are
- Huber loss: MSE for small errors, MAE for large ones (outlier-robust) — deferred as a future improvement

### `run_walk_forward`

```python
def run_walk_forward(train_val_splits, keep_columns, label, model_class, **kwargs) -> list[float]:
    val_scores = []
    for i, (train_dates, val_dates) in enumerate(train_val_splits):
        print(f"Running split {i+1}/{len(train_val_splits)}")
        x_train, y_train = load_split(train_dates, keep_columns, label)
        x_val, y_val = load_split(val_dates, keep_columns, label)
        model = train_model(x_train, y_train, model_class, **kwargs)
        val_score = evaluate_model(model, x_val, y_val)
        print(f"Validation MSE for split {i+1}: {val_score}")
        val_scores.append(val_score)
    return val_scores
```

**Why per-split scores matter:** The average validation MSE tells you typical performance. Per-split scores tell you *where* the model does well and where it underperforms — essential for detecting regime changes or degradation over time.

### Entry Point and Final Test

```python
def entry_point(input_base, val_days, step_days, test_days, keep_columns, label, model_class, **kwargs):
    train_val_splits, test_dates = create_splits(input_base, val_days, step_days, test_days)
    
    val_scores = run_walk_forward(train_val_splits, keep_columns, label, MODELS[model_class], **kwargs)
    print(f"Average validation MSE across splits: {sum(val_scores) / len(val_scores)}")
    
    # Train final model on ALL non-test data
    all_dates = list_dates(input_base)
    train_all = [d for d in all_dates if d not in test_dates]
    full_model = train_model(*load_split(train_all, keep_columns, label), MODELS[model_class], **kwargs)
    
    # Evaluate final model on held-out test set
    x_test, y_test = load_split(test_dates, keep_columns, label)
    test_score = evaluate_model(full_model, x_test, y_test)
    print(f"Test MSE: {test_score}")
```

**Why train a fresh final model (not reuse last split's model):** `run_walk_forward` trains one model per split — the last split's model was trained only on that split's `train_dates`, not all non-test data. The final model should use all available non-test data to give the best possible estimate of production performance.

**Getting all non-test dates:**
```python
train_all = [d for d in all_dates if d not in test_dates]
```
List comprehension preserves chronological order (unlike set subtraction which is unordered). Order matters for time series data.

**`*load_split(...)` unpacking:** `load_split` returns `(X, y)`. `*` unpacks this tuple as the first two positional arguments to `train_model`. Clean and avoids intermediate variables.

### Test Suite: `test_training_pipeline.py`

```python
def test_load_split(tmp_path):
    dir1 = tmp_path / "date=2023-01-01"
    dir1.mkdir()
    df1 = pl.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0],
        "label": [2, 3]
    })
    df1.write_parquet(dir1 / "data.parquet")
    x, y = tp.load_split([dir1], ["feature1", "feature2"], "label")
    assert x.shape == (2, 2)
    assert y.shape == (2,)
    assert (x[:, 0] == [1.0, 2.0]).all()
    assert (x[:, 1] == [3.0, 4.0]).all()
    assert (y == [2, 3]).all()

def test_train_model():
    x_train = [[1.0, 3.0], [2.0, 4.0]]
    y_train = [2, 3]
    model = tp.train_model(x_train, y_train, tp.LinearRegression)
    from sklearn.utils.validation import check_is_fitted
    assert model.coef_.shape == (2,)
    check_is_fitted(model)

def test_evaluate_model():
    from sklearn.linear_model import LinearRegression
    x_val = [[1.0, 3.0], [2.0, 4.0]]
    y_val = [2, 3]
    model = LinearRegression().fit(x_val, y_val)
    mse = tp.evaluate_model(model, x_val, y_val)
    assert mse == 0.0
```

**`tmp_path` fixture:** pytest's built-in fixture providing a real temporary directory that is cleaned up after the test. No imports required — just declare `tmp_path` as a function parameter.

**`check_is_fitted`:** sklearn utility from `sklearn.utils.validation`. Raises `NotFittedError` if the model hasn't been fit. Calling it directly (not inside `pytest.raises`) means: if it doesn't raise, the test passes; if it raises, the test fails.

**`test_evaluate_model` trick:** Evaluating a model on its own training data gives MSE = 0.0 when the model perfectly fits. For a linear regression on 2 points, this is guaranteed. No real data or filesystem access needed.

**Why `run_walk_forward` is not unit tested:** It is integration/glue code — it calls `load_split`, `train_model`, and `evaluate_model`, which are already tested individually. Testing it would essentially test whether Python executes function calls correctly.

---

## Full Test Suite Summary

### `test_book_builder.py` (7 tests)
1. `test_apply_snapshot` — correct structure after snapshot
2. `test_apply_update` — correct structure after updates
3. `test_apply_update_ignored_when_invalid` — updates ignored when book invalid
4. `test_best_price_validation` — invariant violation marks book invalid with reason
5. `test_size_zero` — size 0 removes price level
6. `test_reset` — clears all state including spread history
7. `test_health_check` — anomalous spread marks book invalid (may be removed with health check)

### `test_snapshot_sampler.py` (5 tests)
1. `test_correct_top_bids_asks` — top N bids sorted high→low, asks sorted low→high
2. `test_fewer_levels_than_requested` — missing levels padded with `None`
3. `test_empty_book` — all price/size fields are `None` when book empty
4. `test_correct_key_count` — dict has exactly `N*4 + 3` keys
5. `test_ordering` — bids descending, asks ascending

### `test_snapshot_writer.py` (1 test)
1. `test_correct_shape` — list of N snapshots → DataFrame with N rows and 43 columns, correct partitioned path

### `test_features_pipeline.py`
- One test per feature function with concrete inputs and expected outputs
- Null propagation test
- Rolling spread partial-window test

### `test_label_builder.py`
1. `test_compute_labels` — verifies correct return values with known mid prices, checks null on last row

### `test_validation_splits.py`
1. `test_correct_split` — verifies split structure with `val_days=1, step_days=1`
2. `test_validation_cutoff` — verifies split structure with `val_days=2, step_days=1`
3. `test_insufficient_data_raises` — uses `unittest.mock.patch` to mock `list_dates`, verifies `ValueError` raised

### `test_training_pipeline.py`
1. `test_load_split` — correct X/y shape and values from known input Parquet via `tmp_path`
2. `test_train_model` — fitted model has correct `coef_.shape`, passes `check_is_fitted`
3. `test_evaluate_model` — MSE = 0.0 when evaluated on own training data

### Testing Philosophy
- Pure logic functions (e.g., `train_val_splits`) take plain lists so they can be tested without filesystem
- `tmp_path` pytest fixture used for filesystem-dependent tests
- `unittest.mock.patch` used to test functions that call the filesystem without real data on disk
- Integration/glue code (`run_walk_forward`, `Collector`) intentionally not unit tested — testing them would be testing whether Python executes function calls correctly
