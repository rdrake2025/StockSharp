# NQ Day Trading Desk

A self-contained, single-file web app (`index.html`) for learning and practicing
intraday trading of Nasdaq-100 futures (NQ / MNQ). Open it in any browser, no build step.

| Tab | What it does |
| --- | --- |
| How it works | Day-trading workflow, order book, futures vs. QQQ, a phased learning roadmap |
| NQ futures | NQ/MNQ/ES/MES specs, live front-month and roll calendar, costs, catalysts |
| Strategies | Six setups (ORB, VWAP pullback, VWAP reversion, 9/20 EMA, failed PDH breakout, gap fill) on annotated charts |
| Replay drill | Bar-by-bar simulated session with bracket orders and P&L |
| Risk tools | Position sizer, reward:risk planner, Monte Carlo expectancy simulator |
| Journal | Trade log with P&L after fees, R-multiples, win rate, profit factor, drawdown, by-setup and by-hour breakdowns, CSV copy |
| Game plan | Daily loss/trade limits with a stop-trading gate, pre-market checklist, key levels |

The session clock at the top always shows the current New York time and trading phase.

Journal data is stored in the browser's `localStorage` when opened as a local file.
Strategy and replay charts use simulated prices, not market data. Educational use only; not investment advice.
