# Beginner review: how long until profitable, using only this app

Reviewer persona: brand-new trader (watched some videos, never placed a trade), Eastern time,
free **10:00 am–12:00 pm on weekdays**, $1,330/month spare, LucidFlex 50K plan. Every duration below is either
measured from the app (content size, replay length) or from the 10-year NQ backtest (signal frequency, eval
simulations in **calendar sessions**). "Profitable" here means: **net positive after costs over a rolling
3-month window while following the rules**, first on a sim/eval account.

## 1. Walkthrough log (app v10)

| Step | What I did as the beginner | Time | Friction found |
| --- | --- | --- | --- |
| 1 | Opened Overview, followed "Do it now" to the Learn tab | 5 min | Clear. |
| 2 | Read Learn (≈2,700 words), glossary, visuals, drawdown simulator, quiz | 2–3 h over 2–3 days | Good concepts, but **no step-by-step "how to take a noise-band trade"** on a real chart. The rule lives in the Lab as dense text. |
| 3 | Replay drill, 10 sessions | ~15 min each, ~2.5 h | **Replay defaulted to a 30-pt stop and 60-pt target.** The strategy uses a ≥46-pt stop and a trailing exit, so replay trained the wrong habit. Stepping 72 bars one at a time is slow, when only 12 moments (the checks) matter. |
| 4 | Graded replays (v10) to reach 80% fidelity | 20–40 sessions → **2–4 weeks** | Grading is the right idea, but every repetition costs 10–15 min, so pattern recognition builds slowly. |
| 5 | Tried to plan a real session in the cockpit | — | **Blocking issue:** the proven strategy holds **68% of trades past 12:00** and 35% to the close. With a 10–12 window I **cannot run the strategy as tested**. Nothing in the app said so. |
| 6 | Paper-trade 20 trades through the cockpit (training program weeks 5–6) | Morning signals come on only ~44% of days → **~9 weeks** | The app estimated 2 weeks. The cockpit's noise-band card says LONG/SHORT but doesn't fill in the trade ticket, and the open-trade card doesn't show where to exit. |
| 7 | Tradovate setup | 1–2 h | The checklist names the steps but not **where** to find the open, prior close and VWAP on the platform. |
| 8 | Eval (LucidFlex 50K) | see below | The app said "median 152 days". That counted **only days with a trade**. In calendar sessions the median is **256 at 1 MNQ** (full-day strategy) and **385 for my morning-only version**. |

## 2. Measured facts that drive the timeline

* Morning-only version (entries 10:00–11:30, flat or fixed stop at 12:00): PF 1.26, Sharpe 0.70, t = 2.25,
  8 of 11 years positive, **≈$1,130 a year per MNQ**. It trades on 44% of days.
* Even with perfect execution it is net positive over 1 month **58%** of the time, over 3 months **66%**,
  over 6 months **71%** and over 12 months **72%**. Profit is a months-long statistic, not a daily one.
* LucidFlex 50K, morning version, calendar sessions to pass (median):
  1 MNQ **385** (42% pass, 7% fail) · 2 MNQ **225** (53% pass, 28% fail) ·
  **step-up 1→2 MNQ at +$1,000: 307** (50% pass, 10% fail).

## 3. Timeline before changes (v10, as a 10–12 trader)

| Phase | Duration |
| --- | --- |
| Learn the basics | 1 week |
| Replay to 80% fidelity | 3–4 weeks |
| Paper trading, 20 trades | ~9 weeks (and the strategy didn't fit the schedule) |
| Setup | 1 week |
| Eval at 1 MNQ | ~18 months median, only 42% pass |
| **First 3-month profitable stretch, following rules** | **realistically never, as planned.** The schedule conflict alone breaks it. If fixed ad hoc: ~12–18 months. |

## 4. What the app needs (changes made in v11)

1. **Personal strategy that fits the schedule:** "morning + hand-off". Trade the 10:00–11:30 checks; at 12:00
   put a fixed stop at the trailing level and flatten from a phone by 3:55. Backtested (numbers above) and wired
   into the plan, cockpit, replay grading and rules.
2. **Signal Drill:** rapid-fire chart questions at real check moments ("long, short or no trade?") with
   instant answers. About 20 decisions in 5 minutes vs about 1 in a 15-minute replay, which builds recognition roughly 10× faster.
3. **Replay fixes:** stop defaults to the data-backed safe stop, no fixed target, a "System exits" mode that
   manages trades like the strategy, a "Next check ▸▸" jump, and grading that only counts signals inside my window.
4. **Cockpit:** "Use this signal" pre-fills the trade ticket (side, entry, safe stop). The open-trade card shows
   the live exit level and the 12:00 hand-off instruction.
5. **Strategy in 6 steps** walkthrough on a real day, plus a **Tradovate setup guide** (where each number lives).
6. **Honest calendar time** everywhere (sessions, not trade days), **step-up sizing** in the plan, and a
   **timeline card** that projects dates from my real progress.
7. Training program re-paced: the drill replaces most slow replays, and 10 sim trades (not 20) once drill + replay
   fidelity prove the skill.

## 5. Timeline after changes

| Phase | Before | After |
| --- | --- | --- |
| Learn basics + "strategy in 6 steps" | 1 week | 3–4 days |
| Pattern recognition (drill ≥90% ×3 sets + 5 graded replays ≥80%) | 3–4 weeks | 1–2 weeks |
| Sim trading (10 cockpit trades, morning version) | ~9 weeks | ~4–5 weeks |
| Setup (with Tradovate guide) | 1 week | 2 days (in parallel with sim) |
| Eval, step-up sizing | ~18 months at 1 MNQ | **~15 months median (307 sessions)**, 10% fail |
| **First profitable 3-month stretch following rules** | not reachable as planned | **~3 months after starting the eval, about 2/3 likely**, so ~5 months from day 1 |

The honest ceiling: the app can shrink the **learning** part from months to weeks. It can't make the market pay
faster than the edge does. That's about $1,130 a year per MNQ for a 10–12 schedule, and a lot more
(≈$2,200 per MNQ) only if you can manage trades into the afternoon.
