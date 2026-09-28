# NQ Day Trading Desk

A self-contained web app for learning, backtesting and practicing intraday trading of
Nasdaq-100 futures (NQ / MNQ). Open `index.html` in any browser.

| Tab | What it does |
| --- | --- |
| My plan | Personal guide generated from your profile: hours in your time zone, eval size and contracts, daily stop, budget, rules, roadmap with auto progress, readiness gate |
| Cockpit | Side-by-side live trading panel: next-event countdown with sound, GO/STOP gate, trade ticket that blocks trades failing risk checks, LucidFlex eval tracker (max-loss distance), cool-offs after losses, tilt check. Also: Live New York session clock, front-month contract, trade/no-trade gate from your daily limits, noise-band signal calculator, checklist, key levels |
| Research fleet | Eight research agents modeled on the AnswerRank pipeline (Scout, Auditor, Skeptic, Regime analyst, Portfolio builder, Risk officer, Reporter, Bookkeeper), regime heatmap, portfolios, correlations, Scout research ledger |
| Coach | Skill ladder, coach notes from your journal and graded replays, replay fidelity chart, 8-week training program, funded reality check |
| Prop eval | Lucid Trading LucidFlex rules, pass-rate simulator by size and contracts, luck baseline, how to get the eval |
| Edge tools | Official FOMC / CPI / jobs-report calendar (auto news blackouts), similar-day pre-market outlook with walk-forward skill test, intraday range projection, noise-band exit manager, stop-placement analyzer from real trade excursions |
| Backtest Lab | 41 strategies + benchmark tested on 2,630 real NQ sessions (2015 to Jul 2025): leaderboard, equity curves, yearly results, real example trades |
| Learn | Beginner course: 40-term glossary, candlestick / ticks-to-dollars / bracket / expectancy visuals, interactive trailing-drawdown simulator, consistency-rule checker, why evals fail, 8-question quiz; plus workflow, order book, futures vs. QQQ, contract specs and roll calendar, costs, learning path |
| Replay | Trade 160 real NQ sessions bar by bar with a bracket (keys: B, S, F, →, Space, N) |
| Risk tools | Position sizer, reward:risk planner, Monte Carlo simulator that loads any backtested strategy |
| Journal | Trade log, stats, your results vs. the backtest per setup, CSV copy/import |

## Backtest headline (27 strategies, 0.75 NQ pts cost per round trip, conservative fills)

| Strategy | Trades | Win % | PF | Sharpe | Adj. p (45 variants) | Random-direction p | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Noise-boundary momentum (Zarattini, Aziz & Barbon 2024) | 2,357 | 39% | 1.30 | 1.12 | 0.004 | <0.001 | **Proven** (7/7 tests) |
| Opening drive continuation | 1,275 | 40% | 1.19 | 0.65 | 0.48 | 0.006 | Promising |
| Lunch-range breakout | 2,215 | 49% | 1.17 | 0.64 | 0.50 | 0.004 | Promising |
| Benchmark: long every open→close | 2,630 | 54% | 1.04 | 0.32 | — | — | — |
| 9 fade / pullback / reversion rules | | | < 1.0 | < 0 | | | Loses money |

Walk-forward (settings picked only from the prior 3 years): noise-band momentum was up 7 of 8 years (+12,287 pts);
picking the top 3 of all 27 strategies each year by past Sharpe was up 8 of 8 years (+6,060 pts).

## Lucid Trading (LucidFlex) evaluation simulator

`backtest/lucid.py` replays each strategy through LucidFlex rules (EOD trailing max loss enforced intraday, locks at
start + $100; 50% consistency; min 2 days; funded payouts after 5 qualifying days, 50% of profit up to the cap, 90/10 split),
starting on every third session 2015–2025. On a 50K with noise-band momentum: 1 MNQ passed 66% (0% failed, median 152
sessions); 2 MNQ passed 54% (median 70); 4 MNQ passed 43% (median 30). The same trades with random directions passed 8–13%
and had negative expected value at every size. Rules sourced from Lucid's help center and third-party trackers (Sept 2026).

## Rebuilding

```
# 1. download data (Hugging Face: mdelcristo/NQ-F_1min_OHLCV_Parquet) into $NQ_DATA as NQ_<year>.parquet
cd backtest
NQ_DATA=/path/to/data python3 load.py     # builds per-session arrays
NQ_DATA=/path/to/data python3 run_all.py      # runs all 41 strategies + benchmark (incl. strategies_scout.py)
NQ_DATA=/path/to/data python3 regime.py       # regime analyst + portfolio builder
NQ_DATA=/path/to/data python3 robust.py       # parameter and cost sensitivity
NQ_DATA=/path/to/data python3 proof.py        # significance, bootstrap, randomization, walk-forward
NQ_DATA=/path/to/data python3 lucid.py        # LucidFlex evaluation simulator
NQ_DATA=/path/to/data python3 lucid_luck.py   # random-direction baseline for the simulator
NQ_DATA=/path/to/data python3 funded.py       # funded-account year simulations by size, contracts and payout policy
NQ_DATA=/path/to/data python3 edge.py         # outlook features, range tables, adverse-excursion quantiles
NQ_DATA=/path/to/data python3 export.py       # writes ../bt-data.json
cd .. && python3 build.py                 # src/app.html + bt-data.json -> index.html
```

The data is a community dataset, not verified against CME records. Past results don't predict future
results. Educational use only; not investment advice.
