# Trading Bot

A sequential workspace. **v0.1** downloads, normalizes and validates historical
Binance Spot klines. **v0.2** adds a deterministic indicator layer on top of
that validated store. **v0.3** evaluates a LONG-only entry hypothesis on the
enriched candles and emits signals. **v0.4** replays those signals through a
historical execution simulator. There is no paper trading, no live execution
and no API key anywhere in the project.

## What v0.1 does

Given a symbol, interval and date range in `config/data.yaml`, the pipeline:

1. works out which official Binance archives cover the period;
2. downloads any that are missing, verifying each against its published
   SHA-256 checksum;
3. stores them unmodified in a raw archive directory;
4. reads them without trusting their CSV column names;
5. normalizes every timestamp to timezone-aware UTC and every price and volume
   to an explicit numeric type;
6. writes the result as Parquet;
7. validates the stored dataset and writes a human-readable report;
8. exits `0` for PASS or `1` for FAIL.

Default target: `BTCUSDT`, Binance **Spot**, `5m`, from `2024-01-01` to
`2025-12-31` inclusive — 210,528 expected candles.

## Where the data comes from

The only source is the official
[Binance Public Data](https://github.com/binance/binance-public-data) archive
at `https://data.binance.vision`. No Kaggle, no GitHub datasets, no third-party
mirrors, and no Binance REST or trading API.

Archives are addressed by path:

```
{base_url}/{market}/{monthly|daily}/{data_type}/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{PERIOD}.zip
```

For example:

```
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2024-01.zip
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2024-01.zip.CHECKSUM
```

This template lives in exactly one place, `ARCHIVE_URL_TEMPLATE` in
[`src/data/downloader.py`](src/data/downloader.py), and the base URL itself is a
config value.

## Project structure

```
trading-bot/
├── config/
│   └── data.yaml              Every tunable parameter: symbol, period, paths, retries
├── data/
│   ├── raw/binance/           Untouched Binance ZIPs + their .CHECKSUM sidecars
│   │   └── spot/BTCUSDT/5m/
│   ├── processed/             Normalized, validated Parquet
│   │   └── BTCUSDT/5m/
│   └── downloads/             Staging area for in-flight *.part transfers
├── reports/
│   ├── data_validation_BTCUSDT_5m.txt   Human-readable validation report
│   └── gaps_BTCUSDT_5m.csv              Machine-readable gap list
├── scripts/
│   └── download_btcusdt.py    CLI entry point
├── src/data/
│   ├── downloader.py          Archive planning, retries, checksum verification
│   ├── parser.py              ZIP/CSV reading, positional field mapping
│   ├── normalizer.py          Epoch-unit detection, UTC conversion, dtypes
│   ├── validator.py           All checks; reports problems, never repairs them
│   └── pipeline.py            Orchestration, Parquet storage, report rendering
├── tests/data/                Offline pytest suite
├── requirements.txt
└── README.md
```

### What each module does

| Module | Responsibility |
| --- | --- |
| `downloader.py` | Decides which monthly/daily archives cover the range, downloads what is missing, verifies SHA-256, publishes atomically into `data/raw` |
| `parser.py` | Opens each ZIP, detects whether the CSV has a header, maps the 12 documented fields **by position** onto the internal schema |
| `normalizer.py` | Detects each archive's epoch unit, converts to timezone-aware UTC, casts prices and volumes to explicit numeric types |
| `validator.py` | Checks timestamps, gaps, duplicates, OHLC relations, prices and volumes; returns a report with per-finding severity |
| `pipeline.py` | Wires the stages together, writes Parquet and the reports, produces the PASS/FAIL verdict |

## Raw vs processed

**Raw** (`data/raw/binance/spot/BTCUSDT/5m/`) is the archive exactly as Binance
published it. Nothing in the pipeline ever rewrites a file here. Each `.zip`
sits beside its `.zip.CHECKSUM` sidecar, so the store can be re-verified offline
on later runs. If you delete this directory the pipeline will simply download it
again.

**Processed** (`data/processed/BTCUSDT/5m/`) is derived data: one Parquet file per
source archive, holding the normalized internal schema. It is disposable — it can
always be rebuilt from raw.

Parquet is the primary processed format because it preserves the column types
and the UTC timezone, compresses well, and can be read back column-wise without
parsing text. CSV is used only for the small gap report, never as the processed
store.

### Internal schema

| Column | Type | Notes |
| --- | --- | --- |
| `timestamp` | `datetime64[ns, UTC]` | Candle open time; the ordering key for all sequential processing |
| `open`, `high`, `low`, `close` | `float64` | |
| `volume` | `float64` | Base asset volume |
| `close_time` | `datetime64[ns, UTC]` | |
| `quote_asset_volume` | `float64` | |
| `number_of_trades` | `int64` | |
| `taker_buy_base_asset_volume` | `float64` | |
| `taker_buy_quote_asset_volume` | `float64` | |

Binance publishes a twelfth field that its own documentation labels `Ignore`.
It is the only field not carried into the processed schema, and it is not dropped
blindly: the normalizer verifies the column contains nothing but zeros and raises
if it ever does not.

## Setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux use `python3 -m venv .venv` and `.venv/bin/python`.

## Running the pipeline

```powershell
.\.venv\Scripts\python.exe scripts\download_btcusdt.py
```

Options:

```powershell
.\.venv\Scripts\python.exe scripts\download_btcusdt.py --config config\data.yaml
.\.venv\Scripts\python.exe scripts\download_btcusdt.py --quiet
```

Exit codes: `0` PASS, `1` FAIL (validation found integrity problems), `2` the
run could not complete (bad config, network exhausted, unreadable archive).

To change symbol, interval or period, edit `config/data.yaml`. Nothing needs to
change in Python.

## Running the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite is fully offline. It builds small synthetic datasets — including real
ZIP archives with real checksums — so it never contacts Binance and needs no
credentials. Network behaviour is exercised through a stub HTTP session.

## The validation report

After every run:

- `reports/data_validation_BTCUSDT_5m.txt` — the human-readable report
- `reports/gaps_BTCUSDT_5m.csv` — the gap list as CSV, written even when empty

The report always shows, explicitly:

```
Symbol, Market, Interval, Configured period
Expected candles      (computed from the config, never hardcoded)
Actual candles
Unique timestamps
Missing candles
Duplicates
Invalid OHLC
Invalid prices
Invalid volume
Timestamp errors
First candle / Last candle
Expected first / Expected last / Boundary check
Gap report            (interval, totals, and one row per gap)
Source archives       (epoch units seen, header styles seen, downloads vs cache)
Findings              (each tagged FAIL or WARNING)
STATUS: PASS | FAIL
```

### Severity model

Missing candles are a **WARNING**. Genuine Binance exchange downtime leaves real
holes in the archive, so a gap is something the pipeline must surface rather
than something that invalidates the dataset. Each gap is reported with the
timestamp before it, the timestamp after it, its duration, and the number of
absent candles.

Everything else is a **FAIL**:

- duplicate timestamps
- invalid OHLC relations
- non-positive prices or negative volumes
- timestamps off the interval grid or outside the configured range
- unsorted timestamps
- a missing or non-UTC timezone
- an absent first or last expected candle

Status is PASS when no FAIL finding exists, even if warnings were raised.

### What the validator checks

**Timestamps** — present, unique, sorted ascending, timezone-aware UTC, aligned
to the configured interval, inside the configured range, and containing the exact
first and last expected candle.

**Gaps** — the expected timestamp index is compared against what is stored, and
every contiguous run of absent candles is listed individually.

**OHLC** — `high >= max(open, close)`, `low <= min(open, close)`, `high >= low`.

**Values** — `open`, `high`, `low`, `close` are all strictly positive;
`volume`, the quote and taker volumes, and `number_of_trades` are all
non-negative.

The validator never modifies data. It has no repair path at all: no gap filling,
no interpolation, no forward filling, no de-duplication, no clamping of
suspicious values.

## Architectural decisions

**Epoch unit is detected, not assumed.** Binance Spot archives switched from
millisecond to microsecond timestamps on 2025-01-01, so the default range spans
both. Every unit Binance has published is tested against each archive and the one
that decodes to a plausible calendar year wins; adjacent units differ by three
orders of magnitude, so exactly one ever fits. The result is then cross-checked
against the calendar period the archive's filename advertises, which turns
detection into a *checked* conversion — a misread unit raises instead of
producing plausible nonsense.

**Fields are mapped by position, not by name.** Newer archives ship a CSV header
row and older ones open straight on data. The parser decides by testing whether
the first cell parses as an integer, discards any header, and applies its own
schema from the documented field order.

**One Parquet file per source archive.** Idempotency is structural rather than
bookkept: re-running overwrites exactly the periods it re-derives, so candles
cannot accumulate twice. Parquet files outside the configured range are pruned,
so the processed store always mirrors the config. A single combined file would
need a full rewrite plus a de-duplication pass on every run.

**Validation reads back from Parquet.** The validator runs against what was
actually persisted, not the in-memory frames, so a dtype or timezone lost in the
round-trip would be caught.

**Checksums instead of heuristics.** Every archive has a published
`.zip.CHECKSUM` sidecar, so "is this file already fully downloaded?" is answered
by SHA-256 rather than by inspecting file size. Transfers stage through
`data/downloads` as `*.part` and are published with an atomic rename, so an
interrupted run cannot leave a truncated archive that looks complete.

**Monthly archives preferred, daily as fallback.** Whole calendar months use one
monthly archive; partially covered months fall back to daily files. A two-year
backfill is therefore 24 requests instead of 731.

**`float64` for prices, not `Decimal`.** At BTC magnitudes `float64` is exact
well past the eight decimal places Binance publishes, and it keeps the Parquet
schema simple and operations vectorized. If a future version needs exact decimal
arithmetic for accounting, that belongs in the execution layer, not the data
store.

**Row order is preserved.** The normalizer does not sort. Re-ordering rows would
hide a genuine upstream ordering fault from the validator, and the validator's
job is to detect exactly that.

**Processed Parquet is written even on FAIL.** Normalization is faithful to the
source, so the stored data is worth keeping for inspection. The report and the
exit code carry the verdict rather than the presence or absence of files.

## Idempotency

Running the pipeline twice produces the same result. The second run:

- re-verifies the raw store from local checksum sidecars and downloads nothing;
- rewrites each Parquet file with identical content;
- produces an identical report apart from its `Generated:` line;
- adds no duplicate candles.

This is covered by tests in
[`tests/data/test_pipeline.py`](tests/data/test_pipeline.py).

## Known characteristics of Binance Public Data

### Observed in the default range

- **The epoch unit changes mid-history.** Confirmed directly: all twelve 2024
  archives publish millisecond timestamps and all twelve 2025 archives publish
  microseconds, matching Binance's documented 2025-01-01 change. The report's
  `Epoch units seen` line records this per run. Assuming milliseconds would have
  placed every 2025 candle in January 1970.
- **Month lengths vary and are not padded.** Archives contain 8,928 candles for
  31-day months, 8,640 for 30-day months, 8,352 for February 2024 and 8,064 for
  February 2025. The expected-count check is therefore computed from the
  calendar, not from a per-month constant.
- **No gaps, duplicates or OHLC violations were found** in BTCUSDT 5m across
  2024-01-01 to 2025-12-31. All 210,528 expected candles are present.

### Handled defensively

- **CSV headers are inconsistent across the archive.** Newer archives are
  documented to ship a header row while older ones open straight on data. Every
  archive in this particular range turned out to be headerless (`CSV header
  rows: absent` in the report), but the parser still sniffs each file
  individually and maps fields by position, so a header appearing in a future
  archive cannot shift the columns.
- **Candle close times carry the source's precision.** A millisecond archive
  closes a candle at `:59.999`, a microsecond archive at `:59.999999`. Both
  describe the same candle; neither is rounded away.
- **Monthly archives appear on a delay.** Daily data lands the next UTC day and
  monthly data on the first Monday of the following month, so a range that
  extends to the current month may have no monthly archive yet.
- **Not every period exists.** Periods before a symbol was listed return HTTP
  404. These are recorded as "not published" rather than treated as errors.
- **Archives are occasionally re-published.** Binance maintains a changelog of
  corrected files. Because the pipeline always re-derives processed data from
  raw and verifies checksums, replacing a raw archive and re-running is enough
  to pick up a correction.
- **Real gaps can exist.** Exchange downtime produces genuinely missing candles.
  This is why missing candles are a warning, not a failure.

## Limitations of v0.1

- One symbol, one market and one interval per run.
- Spot klines only. Other data types (`aggTrades`, `trades`) and other markets
  (`um`, `cm`) are reachable through config but untested.
- Calendar-month intervals (`1mo`) are rejected, since they have no fixed
  duration.
- Downloads are sequential. For a two-year 5m backfill this is 24 files and
  takes well under a minute, so concurrency would add complexity for no
  meaningful gain.
- Gaps are detected and reported but never filled. There is deliberately no
  repair path.
- No incremental "append only the newest candles" mode; a run always re-derives
  the whole configured range from raw. This is cheap at this scale and removes a
  whole class of state-drift bugs.

## Out of scope, deliberately

v0.1 contains no trading logic: no buy or sell decisions, no stop loss, no take
profit, no position sizing. No backtesting engine. No paper or live trading. No
dashboard. No Binance API integration and no API keys — the data pipeline only
performs anonymous HTTP GETs against the public data archive.

The schema is built for sequential, look-ahead-free processing: candles are keyed
by open time in ascending order, and each row contains only information known by
its own close time.

## Forward constraint

When risk management, position sizing and execution are added in a later
version, the project's starting capital is **$20**. No other starting-capital
figure should appear in code, configuration or examples unless that requirement
is explicitly changed. v0.1 uses no capital at all, so this value intentionally
appears nowhere in `config/data.yaml`.

---

# v0.2 — Indicator Engine

v0.2 computes four look-ahead-free indicators on the validated v0.1 OHLCV
store. It does **not** generate signals, entries, exits, position sizes, or
P&L. There is still no strategy, no backtester, and no Binance trading API.

## Purpose

Take the processed BTCUSDT 5m dataset, append `ema20`, `ema50`, `rsi14` and
`volume_ma20`, and write the result to a separate enriched Parquet tree so the
original processed files never change.

## Supported indicators

Periods come from [`config/indicators.yaml`](config/indicators.yaml), not from
Python literals in `engine.py`.

| Column | Definition |
| --- | --- |
| `ema20` / `ema50` | Exponential moving average of **close**. `alpha = 2 / (period + 1)`. The first valid value is the SMA of the first `period` closes; then the recursive EMA formula. |
| `rsi14` | Wilder RSI of **close**. Seeded with the SMA of the first 14 gains and 14 losses, then Wilder-smoothed. |
| `volume_ma20` | Simple moving average of **volume**, current bar included. Not exponential. |

Formulas are implemented in [`src/indicators/ema.py`](src/indicators/ema.py),
[`rsi.py`](src/indicators/rsi.py) and [`volume.py`](src/indicators/volume.py).
`engine.py` only copies the frame and composes those functions. No TA library
is used.

## Warm-up

Leading NaNs are expected and are **not** filled with 0 or 50.

| Column | Leading NaNs (when the source has no holes) | First valid index |
| --- | --- | --- |
| `ema20` | 19 | 19 (SMA of closes `[0:20]`) |
| `ema50` | 49 | 49 |
| `rsi14` | 14 | 14 (needs 14 price *changes*, so 15 closes) |
| `volume_ma20` | 19 | 19 (mean of volumes `[0:20]`) |

RSI special cases, applied only after warm-up: `avg_loss == 0` and gain > 0 →
100; `avg_gain == 0` and loss > 0 → 0; both 0 → 50. Warm-up NaNs stay NaN.

## No look-ahead

Every value at row `t` uses only `close`/`volume` at rows `≤ t`. Tests mutate a
future row and assert that every earlier indicator value is unchanged.

Calculations never read `timestamp`, never inspect calendar months, and never
use future rows. A month boundary is a **storage** boundary only.

## Processed vs enriched

**Processed** (`data/processed/BTCUSDT/5m/`) remains the v0.1 source of truth.
The enrichment pipeline reads it and does not write it.

**Enriched** (`data/enriched/BTCUSDT/5m/`) is derived: the same monthly file
names, plus the four indicator columns. Extra v0.1 fields (`close_time`, quote
and taker volumes, trade count) pass through unchanged.

## Month-boundary continuity

Indicators are computed once on the concatenated chronological dataset, then
split into monthly Parquet files. EMA/RSI/SMA on `2024-02-01 00:00` continue
from January; they are not re-seeded at the month start. A two-month synthetic
test checks that February values from the full series differ from a
February-only reset, and that the written February file matches the full series.

## How to run the indicator pipeline

Requires the v0.1 processed store to already exist (do not re-download):

```powershell
.\.venv\Scripts\python.exe scripts\enrich_btcusdt.py
```

```powershell
.\.venv\Scripts\python.exe scripts\enrich_btcusdt.py --quiet
```

Exit codes match v0.1: `0` PASS, `1` FAIL, `2` configuration or I/O error.

Report: `reports/indicator_validation_BTCUSDT_5m.txt`.

## How to run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

v0.2 tests live in `tests/indicators/` and use small synthetic series. They do
not contact Binance. Formula tests encode expected values by hand; they do not
call the function under test to obtain the expected result.

## v0.2 contains no trading logic

The indicator engine does not decide BUY or SELL. It does not implement
entries, exits, stop loss, take profit, position sizing, paper trading, or live
trading. Invalid input is not repaired: `calculate_indicators` fails if
`close`/`volume` are not already floating-point columns. Data validation stays
in v0.1.

---

# v0.3 — Strategy Engine

v0.3 reads the v0.2 enriched store and asks, for each already-closed 5m
candle: does the configured hypothesis generate a LONG entry signal? It does
**not** size positions, apply stop loss or take profit, simulate fills, or
compute P&L. There is still no backtester and no Binance trading API.

## Hypothesis

Market: BTCUSDT, Binance Spot, 5m, LONG only. All four conditions must be
true after candle N closes:

1. Fast EMA > slow EMA (strict), using the v0.2 columns.
2. Close of N is strictly greater than the highest high of the previous
   `breakout_period` **closed** candles (N-period through N-1). The current
   candle's high is never in that window.
3. Volume is strictly greater than `volume_ma * volume_multiplier`.
4. RSI is strictly inside `(rsi_min, rsi_max)`.

AND logic. Equalities fail. NaN indicator values fail closed: no signal, no
fill-in. Parameters live in [`config/strategy.yaml`](config/strategy.yaml).
Indicator *periods* stay in [`config/indicators.yaml`](config/indicators.yaml)
and are not duplicated.

`breakout_period` is intentionally independent of `volume_ma_period`, even
when both happen to be 20. One is the previous-high lookback; the other is
the v0.2 volume SMA length. They must not be coupled.

## Signal timing

`timestamp` is candle N's **open** (the dataset key). Evaluation happens
after that candle closes. `earliest_execution_time` is the next interval
boundary. `reference_close` is the closed candle's close, for audit only —
not a fill. The engine never stores an execution price, fill price, or
entry price.

The last historical candle may emit `LONG_ENTRY`. Its
`earliest_execution_time` may point past the dataset. In v0.3 that field is
only a timing label: no synthetic next candle and no fill.

The engine is stateless. Consecutive `LONG_ENTRY` rows are allowed. One
position at a time belongs to a future portfolio layer.

## How to run

Requires the v0.2 enriched store to already exist:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_strategy.py
```

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_strategy.py --quiet
```

Exit codes match v0.1: `0` PASS, `1` FAIL, `2` configuration or I/O error.

Reports:

- `reports/strategy_validation_BTCUSDT_5m.txt` — counts, warm-up, condition
  failures, integrity. No P&L.
- `reports/strategy_signals_BTCUSDT_5m.csv` — `LONG_ENTRY` rows only.

Processed and enriched Parquet are not written.

## How to run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

v0.3 tests live in `tests/strategy/` and use small synthetic enriched
frames. They do not contact Binance and they do not recalculate indicators.

## v0.3 contains no execution or risk

The strategy engine does not place orders, simulate fills, apply commissions
or slippage, compute position size, or implement stop loss / take profit.
Those belong to later components. Invalid input is not repaired.

---

# v0.4 — Backtesting Engine

v0.4 replays Strategy #1 sequentially on the v0.2 enriched store and asks:
if we had traded those signals under the current experimental execution and
risk assumptions, which fills would have occurred? It does **not** optimize
the strategy, add indicators, or talk to Binance.

`STATUS: PASS` means the simulator matched the specified v0.4 integrity
rules and passed its validation tests. It does **not** mean:

- the strategy is profitable
- the strategy has positive expectancy
- the strategy is statistically significant
- the strategy is suitable for live trading
- the strategy should be deployed
- the strategy has been optimized

## Execution model

Signals come from v0.3 unchanged. The strategy stays stateless. Consecutive
`LONG_ENTRY` rows are still valid; the portfolio layer allows only one LONG
at a time.

- Entry: signal on candle N fills at candle N+1 **OPEN**, then 0.05% adverse
  slippage. No fill at N close.
- Sizing happens at that fill, from current cash, the execution OPEN, the
  configured entry fee, and the 1% stop. Quantity is never frozen when the
  pending entry is scheduled.
- SL is 1% below the **actual entry fill**; TP is 2% above it.
- Same-candle SL and TP: assume SL first. Equality triggers.
- Fees: 0.1% entry and 0.1% exit (experimental assumptions, not a live
  account fact).
- Starting capital: **$20**. Risk per trade: 1% of cash at fill time.
- Maximum holding time / `TIME_EXIT` is not part of this baseline.
- Open at end of data: `OPEN` / `END_OF_DATA`. No invented exit, no exit
  fee, no exit slippage, no realized P&L. Informational mark-to-market uses
  the last close and is not a fill.

## How to run

Requires the v0.2 enriched store to already exist:

```powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py
```

```powershell
.\.venv\Scripts\python.exe scripts\run_backtest.py --quiet
```

Exit codes match earlier milestones: `0` PASS, `1` FAIL, `2` configuration
or I/O error.

Reports:

- `reports/backtest_validation_BTCUSDT_5m.txt` — engine integrity, not
  profitability
- `reports/backtest_trades_BTCUSDT_5m.csv` — trade ledger
- `reports/backtest_ignored_signals_BTCUSDT_5m.csv` — rejected signals

Processed and enriched Parquet are not written. Risk and execution
parameters live in [`config/backtest.yaml`](config/backtest.yaml).

## How to run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

v0.4 tests live in `tests/backtest/` and use small synthetic frames first.
They do not contact Binance and they do not recalculate indicators.

## v0.4 contains no live execution or analytics framework

The backtest does not place orders, does not use API keys, does not paper
trade, and does not compute Sharpe, Sortino, walk-forward, or Monte Carlo
statistics. Those belong to later components. Invalid input is not repaired.

