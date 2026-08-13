# Data directory

| File | What it is |
|---|---|
| `xauusd_sample.csv` | **Synthetic** daily XAUUSD data (782 rows, 2023-08-10 .. 2026-08-07).<br>Bundled into the one-click EXE as an **offline fallback** so the pipeline can always complete stage 3/4 without network.<br>⚠️ It is randomly generated - training/backtest results on it have **no real-world meaning**. For real results run `python data_fetch.py` (Yahoo Finance) or drop your own `xauusd_data.csv` next to the exe. |

Real data takes precedence automatically: `run_all.py` stage 3 uses, in order,
`xauusd_data.csv` next to the exe → fresh Yahoo download → this sample.
