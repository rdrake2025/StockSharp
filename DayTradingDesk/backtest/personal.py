"""Personal version of noise-band momentum for a 10:00-12:00 ET schedule ("morning + hand-off"):
entries only at the 10:00-11:30 checks, normal trailing exits until 12:00, then a fixed stop at the 12:00 trailing
level rides unattended to the close. Produces eval (calendar sessions), step-up sizing and funded results."""
import os, pickle, json, math
import numpy as np
import bt, lucid, funded, window
from lucid import MICRO_EXTRA

D = os.environ.get("NQ_DATA", "data")
days = pickle.load(open(os.path.join(D, "days.pkl"), "rb")); dates = [str(d["date"]) for d in days]
tr = bt.run(days, window.noise_window_hold(119, 150))
pts, low, n = lucid.day_series(tr, dates)
S = lucid.SIZES["50K"]
starts = list(range(0, len(pts) - 520, 3))

def run(s0, steps, S=S, maxd=500):
    bal = S["start"]; peak = bal; mll = bal - S["mll"]; locked = False; best = 0; t = 0
    for k in range(s0, min(len(pts), s0 + maxd)):
        if n[k] == 0: continue
        t += 1; prof = bal - S["start"]; m = max(c for th, c in steps if prof >= th)
        per = m * 2; cost = MICRO_EXTRA * n[k] * per
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

plans = {"1 MNQ": [(-1e9, 1)], "2 MNQ": [(-1e9, 2)], "Step-up: 1 MNQ, then 2 at +$1,000": [(-1e9, 1), (1000, 2)]}
ev = {}
for nm, st in plans.items():
    rr = [run(s, st) for s in starts]; cal = [x[1] for x in rr if x[0] == "pass"]
    ev[nm] = dict(pass_rate=round(float(np.mean([x[0] == "pass" for x in rr])), 3), fail_rate=round(float(np.mean([x[0] == "fail" for x in rr])), 3),
                  p25=int(np.percentile(cal, 25)), p50=int(np.median(cal)), p75=int(np.percentile(cal, 75)))
    print(nm, ev[nm])
m = bt.metrics(tr, len(days)); o = bt.metrics([t for t in tr if t["date"] >= "2022"], sum(str(d["date"]) >= "2022" for d in days))
yrs = {}
for t in tr: yrs[t["date"][:4]] = yrs.get(t["date"][:4], 0) + t["pts"]
daily = {}
for t in tr: daily[t["date"]] = daily.get(t["date"], 0) + t["ret"]
dr = np.array(list(daily.values()) + [0] * (len(days) - len(daily)))
# probability of being net positive over rolling windows of N sessions at 1 MNQ
cum = np.cumsum(pts * 2 - MICRO_EXTRA * n * 2)
roll = {w: round(float(np.mean(cum[w:] - cum[:-w] > 0)), 3) for w in (21, 63, 126, 252)}
stats = dict(n=m["n"], win=round(m["win"], 3), pf=round(m["pf"], 3), sharpe=round(m["sharpe"], 2), oos_sharpe=round(o["sharpe"], 2),
             up_years=sum(v > 0 for v in yrs.values()), years=len(yrs), per_year_mnq=round(m["tot_pts"] * 2 / 10.55),
             t=round(float(dr.mean() / (dr.std(ddof=1) / math.sqrt(len(dr)))), 2), trade_day_share=round(float((n > 0).mean()), 2),
             avg_win=round(m["avg_win"], 1), avg_loss=round(m["avg_loss"], 1), positive_over=roll)
print(stats)
fd = {}
for sz, mm, pol, c in [("50K", 1, "cushion1500", 1500), ("50K", 2, "cushion1500", 1500), ("100K", 2, "cushion2500", 2500)]:
    rr = [funded.funded_year(pts, low, n, s, mm, lucid.SIZES[sz], c) for s in list(range(0, len(pts) - 250, 3))]
    paid = np.array([x[0] for x in rr])
    fd[f"{sz}|{mm}|{pol}"] = dict(mean=round(float(paid.mean())), p50=round(float(np.median(paid))), p75=round(float(np.percentile(paid, 75))),
                                 breach=round(float(np.mean([x[2] for x in rr])), 3), zero=round(float((paid == 0).mean()), 3))
json.dump(dict(stats=stats, eval=ev, funded=fd, window=json.load(open(os.path.join(D, "window.json")))), open(os.path.join(D, "personal.json"), "w"), default=float)
