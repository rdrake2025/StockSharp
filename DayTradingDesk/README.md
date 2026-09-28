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
| Learn | "Your strategy in 6 steps" on a real NQ morning; beginner course: 40-term glossary, candlestick / ticks-to-dollars / bracket / expectancy visuals, interactive trailing-drawdown simulator, consistency-rule checker, why evals fail, 8-question quiz; plus workflow, order book, futures vs. QQQ, contract specs and roll calendar, costs, learning path |
| Study | Study the market: daily 4-step session, Market Lab (filter 2,630 sessions by weekday, gap, first 30 min, prior day, volatility, Fed/opex days and years; average path, time of high/low, gap fills, trend days, significance tags), read-the-day chart practice, spaced-repetition flashcards, and a pre/post-market notebook |
| Replay | Signal drill (20 rapid-fire "long / short / no trade" questions at real check moments), then trade 160 real NQ sessions bar by bar with a bracket, optional system-managed exits and 12:00 hand-off, a "Next check" jump, and grading against the strategy's signals in your window (keys: B, S, F, →, Space, N) |
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
starting on every third session 2015–2025. On a 50K with noise-band momentum: 1 MNQ passed 66% (0% failed, median 256
calendar sessions); 2 MNQ median 121; 3 MNQ median 70. The same trades with random directions passed 8–13%
and had negative expected value at every size. Rules sourced from Lucid's help center and third-party trackers (Sept 2026).

## Personal version: morning + hand-off (10:00–12:00 ET schedule)

The full strategy holds 68% of trades past noon. `backtest/window.py` and `backtest/personal.py` test a version for
a 10–12 schedule: entries only at the 10:00–11:30 checks, trailing exits until 12:00, then a fixed stop at the
12:00 trailing level held to the close. PF 1.26, Sharpe 0.70 (t = 2.25), 8 of 11 years up, about $1,130 a year per MNQ.
LucidFlex 50K, calendar sessions to pass: 1 MNQ median 385 (42% pass, 7% fail); step-up 1 → 2 MNQ at +$1,000
median 307 (50% pass, 10% fail). `BEGINNER_REVIEW.md` logs a beginner walkthrough of the app, the time to
profitability before and after the v11 changes, and why.

`backtest/aggressive.py` trades the 50K at 1–10 MNQ and buys a new eval after each blow-up. On the 10–12 schedule:
5 MNQ ≈ 2.3 tries, median ~12 months; 10 MNQ ≈ 3 tries, median ~5 months. The worst losing streak in the data is 18 trades.

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
NQ_DATA=/path/to/data python3 window.py       # time-window variants of noise-band
NQ_DATA=/path/to/data python3 personal.py     # morning + hand-off version: stats, eval plans, funded
NQ_DATA=/path/to/data python3 aggressive.py   # bigger eval size with resets: tries, fees, time to pass
NQ_DATA=/path/to/data python3 study.py      # per-session features for the Market Lab
NQ_DATA=/path/to/data python3 edge.py         # outlook features, range tables, adverse-excursion quantiles
NQ_DATA=/path/to/data python3 export.py       # writes ../bt-data.json
cd .. && python3 build.py                 # src/app.html + bt-data.json -> index.html
```

The data is a community dataset, not verified against CME records. Past results don't predict future
results. Educational use only; not investment advice.
