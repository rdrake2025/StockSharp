"""Bigger size in the LucidFlex 50K: losing streaks, and time/cost to first pass when you buy a new eval after each fail."""
import os, pickle, json
import numpy as np
import bt, lucid, window
from lucid import MICRO_EXTRA
D = os.environ.get("NQ_DATA", "data")
days = pickle.load(open(os.path.join(D, "days.pkl"), "rb")); dates = [str(d["date"]) for d in days]
S = lucid.SIZES["50K"]
def run(pts, low, n, s0, m, maxd=500):
    bal = S["start"]; peak = bal; mll = bal - S["mll"]; locked = False; best = 0; t = 0
    for k in range(s0, min(len(pts), s0 + maxd)):
        if n[k] == 0: continue
        t += 1; per = m * 2; cost = MICRO_EXTRA * n[k] * per
        if bal + low[k] * per - cost <= mll: return "fail", k - s0 + 1
        day = pts[k] * per - cost; bal += day
        if bal <= mll: return "fail", k - s0 + 1
        best = max(best, day); peak = max(peak, bal)
        if not locked:
            if peak > S["start"] + S["mll"] + 100: locked = True; mll = S["start"] + 100
            else: mll = peak - S["mll"]
        prof = bal - S["start"]
        if prof >= S["target"] and t >= 2 and best <= .5 * prof: return "pass", k - s0 + 1
    return "timeout", maxd
out = {}
res = pickle.load(open(os.path.join(D, "res_all.pkl"), "rb"))
for nm, tr in [("personal", bt.run(days, window.noise_window_hold(119, 150))), ("full", res["noise"]["trades"])]:
    pts, low, n = lucid.day_series(tr, dates)
    seq = [t["pts"] > 0 for t in sorted(tr, key=lambda t: (t["date"], t["i"]))]
    runs = []; c = 0
    for w in seq:
        c = 0 if w else c + 1; runs.append(c)
    mx = max(runs)
    # risk per trade at 23.5k with 46-pt stop
    out[nm] = {"max_losing_streak": mx, "streak5_times": int(sum(1 for i in range(1, len(runs)) if runs[i] == 5)), "sizes": {}}
    for m in (1, 2, 3, 4, 5, 6, 8, 10):
        tot, att, spent = [], [], []
        for s in range(0, len(pts) - 900, 3):
            k, a = s, 0
            while True:
                a += 1; r = run(pts, low, n, k, m)
                k += r[1]
                if r[0] == "pass": tot.append(k - s); att.append(a); break
                if r[0] == "timeout" or k - s > 800 or k >= len(pts) - 1: tot.append(None); att.append(a); break
        ok = [x for x in tot if x is not None]
        single = [run(pts, low, n, s, m) for s in range(0, len(pts) - 520, 3)]
        out[nm]["sizes"][m] = dict(
            risk_per_trade=round(m * (46 * 2 + 1.24)), pct_of_dd=round(m * (46 * 2 + 1.24) / 2000 * 100, 1),
            one_try_pass=round(np.mean([x[0] == "pass" for x in single]), 3), one_try_fail=round(np.mean([x[0] == "fail" for x in single]), 3),
            pass_within_800=round(len(ok) / len(tot), 3), med_sessions=int(np.median(ok)) if ok else None,
            p25=int(np.percentile(ok, 25)) if ok else None, med_attempts=float(np.median([a for a, x in zip(att, tot) if x is not None])) if ok else None,
            mean_attempts=round(float(np.mean([a for a, x in zip(att, tot) if x is not None])), 2) if ok else None)
        print(nm, m, out[nm]["sizes"][m])
    print(nm, "max losing streak", mx, "5-loss streaks", out[nm]["streak5_times"])
json.dump(out, open(os.path.join(D, "aggressive.json"), "w"), default=float)
