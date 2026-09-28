"""Replay strategies through Lucid Trading's LucidFlex evaluation + funded rules, starting on every historical session.

Rules modeled (Lucid help center + third-party rule trackers, Sept 2026):
  Eval:   profit target; End-of-day trailing Max Loss Limit (MLL) that trails the highest EOD balance and locks at
          start + $100 once the balance exceeds start + MLL + $100; the MLL is enforced intraday (equity touching it = fail);
          no daily loss limit; consistency: largest day <= 50% of total profit at the time of passing; min 2 trading days;
          max contracts per size.
  Funded: same MLL mechanics; payout after 5 profitable days of >= min_day since the last payout; payout = min(50% of
          profit, cap), min $500; the MLL resets to the locked level (start + $100) after a payout; trader keeps 90%;
          scaling plan limits contracts by profit tier.
Costs: backtest trades already pay 0.75 NQ pts; micros pay an extra 0.37 pts per trade (≈ $1.24 round-trip commission on $2/pt).
"""
import os, pickle, json
import numpy as np

SIZES = {
    "25K":  dict(start=25000,  target=1250, mll=1000, max_micro=20,  min_day=100, cap=1000, fee=89,  tiers=[(1000, 10), (1e18, 20)]),
    "50K":  dict(start=50000,  target=3000, mll=2000, max_micro=40,  min_day=150, cap=2000, fee=146, tiers=[(1000, 20), (2000, 30), (1e18, 40)]),
    "100K": dict(start=100000, target=6000, mll=3000, max_micro=60,  min_day=200, cap=2500, fee=293, tiers=[(1000, 30), (2000, 40), (3000, 50), (1e18, 60)]),
    "150K": dict(start=150000, target=9000, mll=4500, max_micro=100, min_day=250, cap=3000, fee=407, tiers=[(1000, 40), (2000, 50), (3000, 60), (4500, 80), (1e18, 100)]),
}
MICRO_EXTRA = 0.37

def day_series(trades, dates):
    """Per session: net pts (1 contract), worst intraday cumulative pts, number of trades."""
    di = {d: i for i, d in enumerate(dates)}
    pts = np.zeros(len(dates)); low = np.zeros(len(dates)); n = np.zeros(len(dates), int)
    by = {}
    for t in trades: by.setdefault(t["date"], []).append(t)
    for d, ts in by.items():
        i = di[d]; cum = 0.0; lo = 0.0
        for t in sorted(ts, key=lambda t: t["i"]):
            lo = min(lo, cum + t["mae"]); cum += t["pts"]
        pts[i] = cum; low[i] = min(lo, cum); n[i] = len(ts)
    return pts, low, n

def eval_run(pts, low, n, s0, micros, S, max_days=500):
    """Returns (outcome, days_used, end_index). outcome: pass / fail / timeout."""
    bal = S["start"]; peak = bal; mll = bal - S["mll"]; locked = False
    best_day = 0.0; traded = 0
    for k in range(s0, min(len(pts), s0 + max_days)):
        if n[k] == 0: continue
        traded += 1
        per = micros * 2.0
        cost = MICRO_EXTRA * n[k] * per
        if bal + low[k] * per - cost <= mll: return "fail", traded, k
        day = pts[k] * per - cost; bal += day
        if bal <= mll: return "fail", traded, k
        best_day = max(best_day, day)
        peak = max(peak, bal)
        if not locked:
            if peak > S["start"] + S["mll"] + 100: locked = True; mll = S["start"] + 100
            else: mll = peak - S["mll"]
        prof = bal - S["start"]
        if prof >= S["target"] and traded >= 2 and best_day <= 0.5 * prof: return "pass", traded, k
    return "timeout", traded, min(len(pts), s0 + max_days) - 1

def funded_run(pts, low, n, s0, micros, S, horizon=250):
    """Trade the funded account for `horizon` sessions. Returns (total paid to trader, payouts, breached)."""
    bal = S["start"]; peak = bal; mll = bal - S["mll"]; locked = False
    good_days = 0; paid = 0.0; npay = 0; traded = 0
    for k in range(s0, min(len(pts), s0 + horizon)):
        if n[k] == 0: continue
        traded += 1
        prof = bal - S["start"]
        lim = next(c for th, c in S["tiers"] if prof < th)
        m = min(micros, lim, S["max_micro"]); per = m * 2.0; cost = MICRO_EXTRA * n[k] * per
        if bal + low[k] * per - cost <= mll: return paid, npay, True
        day = pts[k] * per - cost; bal += day
        if bal <= mll: return paid, npay, True
        if day >= S["min_day"]: good_days += 1
        peak = max(peak, bal)
        if not locked:
            if peak > S["start"] + S["mll"] + 100: locked = True; mll = S["start"] + 100
            else: mll = peak - S["mll"]
        prof = bal - S["start"]
        if good_days >= 5 and prof > 0:
            amt = min(0.5 * prof, S["cap"])
            if amt >= 500:
                bal -= amt; paid += 0.9 * amt; npay += 1; good_days = 0
                locked = True; mll = S["start"] + 100; peak = bal
    return paid, npay, False

def study(series, sizes=("25K", "50K", "100K", "150K"), micro_opts=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20)):
    pts, low, n = series; out = []
    starts = [k for k in range(0, len(pts) - 60) if k % 3 == 0]      # every third session as a start date
    for sz in sizes:
        S = SIZES[sz]
        for m in micro_opts:
            if m > S["max_micro"]: continue
            res = [eval_run(pts, low, n, s, m, S) for s in starts]
            p = np.mean([r[0] == "pass" for r in res]); f = np.mean([r[0] == "fail" for r in res])
            dp = [r[1] for r in res if r[0] == "pass"]
            fund = [funded_run(pts, low, n, r[2] + 1, m, S) for r in res if r[0] == "pass" and r[2] + 1 < len(pts) - 250]
            paid = np.mean([x[0] for x in fund]) if fund else 0.0
            out.append(dict(size=sz, micros=m, pass_rate=round(float(p), 3), fail_rate=round(float(f), 3),
                            med_days=int(np.median(dp)) if dp else None, p75_days=int(np.percentile(dp, 75)) if dp else None,
                            cost_per_pass=round(S["fee"] / p) if p > 0 else None,
                            funded_paid=round(float(paid)), funded_breach=round(float(np.mean([x[2] for x in fund])), 3) if fund else None,
                            ev=round(float(p * paid - S["fee"]))))
    return out

if __name__ == "__main__":
    D = os.environ.get("NQ_DATA", "data")
    days = pickle.load(open(os.path.join(D, "days.pkl"), "rb")); dates = [str(d["date"]) for d in days]
    res = pickle.load(open(os.path.join(D, "res_all.pkl"), "rb"))
    cands = ["noise", "drive", "lunch", "late_mom", "gap_go", "orb5_z", "bench"]
    allout = {}
    for k in cands:
        ser = day_series(res[k]["trades"], dates); allout[k] = study(ser)
        best = sorted(allout[k], key=lambda r: -r["ev"])[:4]
        print(k, [(r["size"], r["micros"], r["pass_rate"], r["med_days"], r["cost_per_pass"], r["funded_paid"], r["ev"]) for r in best])
    # combo: noise + lunch (different times of day)
    combo_tr = res["noise"]["trades"] + res["lunch"]["trades"]
    ser = day_series(combo_tr, dates); allout["noise+lunch"] = study(ser)
    best = sorted(allout["noise+lunch"], key=lambda r: -r["ev"])[:4]
    print("noise+lunch", [(r["size"], r["micros"], r["pass_rate"], r["med_days"], r["cost_per_pass"], r["funded_paid"], r["ev"]) for r in best])
    json.dump(allout, open(os.path.join(D, "lucid.json"), "w"))
    print([(r["size"], r["micros"], r["pass_rate"], r["fail_rate"], r["med_days"], r["funded_paid"], r["ev"]) for r in allout["noise"] if r["size"] == "50K"])
