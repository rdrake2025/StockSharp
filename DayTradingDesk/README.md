# NQ Day Trading Desk

A self-contained web app for learning, backtesting and practicing intraday trading of
Nasdaq-100 futures (NQ / MNQ). Open `index.html` in any browser.

| Tab | What it does |
| --- | --- |
| Today | Live New York session clock, front-month contract, trade/no-trade gate from your daily limits, noise-band signal calculator, checklist, key levels |
| Backtest Lab | 15 strategies + benchmark tested on 2,630 real NQ sessions (2015 to Jul 2025): leaderboard, equity curves, yearly results, real example trades |
| Learn | Workflow, order book, futures vs. QQQ, contract specs and roll calendar, costs, learning path |
| Replay | Trade 160 real NQ sessions bar by bar with a bracket (keys: B, S, F, →, Space, N) |
| Risk tools | Position sizer, reward:risk planner, Monte Carlo simulator that loads any backtested strategy |
| Journal | Trade log, stats, your results vs. the backtest per setup, CSV copy/import |

## Backtest headline (0.75 NQ pts cost per round trip, conservative fills)

| Strategy | Trades | Win % | PF | Sharpe | 2022–25 Sharpe | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Noise-boundary momentum (Zarattini, Aziz & Barbon 2024) | 2,357 | 39% | 1.30 | 1.12 | 1.50 | Edge |
| 5-min ORB, first-candle direction (Zarattini & Aziz 2023) | 2,602 | 23% | 1.10 | 0.42 | 0.56 | Marginal |
| Gap and go (≥0.5% gap, 15-min range break) | 664 | 42% | 1.15 | 0.40 | 0.54 | Edge |
| Benchmark: long every open→close | 2,630 | 54% | 1.04 | 0.32 | 0.34 | — |
| VWAP pullback / VWAP 2σ reversion / 9-20 EMA pullback / failed PDH / last-30-min momentum | | | < 1.0 | < 0 | | No edge |

Momentum and breakout rules beat pullback and fade rules on NQ. Noise-boundary momentum stayed
positive across 9 parameter sets (PF 1.25–1.31) and at costs up to 2 pts per trade (PF 1.21).

## Rebuilding

```
# 1. download data (Hugging Face: mdelcristo/NQ-F_1min_OHLCV_Parquet) into $NQ_DATA as NQ_<year>.parquet
cd backtest
NQ_DATA=/path/to/data python3 load.py     # builds per-session arrays
NQ_DATA=/path/to/data python3 bt.py       # runs all strategies
NQ_DATA=/path/to/data python3 robust.py   # parameter and cost sensitivity
NQ_DATA=/path/to/data python3 export.py   # writes ../bt-data.json
cd .. && python3 build.py                 # src/app.html + bt-data.json -> index.html
```

The data is a community dataset, not verified against CME records. Past results don't predict future
results. Educational use only; not investment advice.
