"""What can a funded LucidFlex account realistically pay? Simulate one year (250 sessions) of the funded phase
from every third historical start date, for several sizes and payout policies.

Payout policy matters because a payout resets the max-loss floor to start + $100:
  asap      withdraw the largest allowed amount as soon as 5 qualifying days are in
  cushionX  only withdraw what keeps the balance at least $X above the floor afterwards
"""
import os, pickle, json
import numpy as np
import lucid
from lucid import SIZES, MICRO_EXTRA

def funded_year(pts, low, n, s0, micros, S, cushion=None, horizon=250):
    bal = S["start"]; peak = bal; mll = bal - S["mll"]; locked = False
    good = 0; paid = 0.0; npay = 0; first = None; traded = 0
    for k in range(s0, min(len(pts), s0 + horizon)):
        if n[k] == 0: continue
        traded += 1
        prof = bal - S["start"]
        lim = next(c for th, c in S["tiers"] if prof < th)
        m = min(micros, lim, S["max_micro"]); per = m * 2.0; cost = MICRO_EXTRA * n[k] * per
        if bal + low[k] * per - cost <= mll: return paid, npay, True, first, traded
        day = pts[k] * per - cost; bal += day
        if bal <= mll: return paid, npay, True, first, traded
        if day >= S["min_day"]: good += 1
        peak = max(peak, bal)
        if not locked:
            if peak > S["start"] + S["mll"] + 100: locked = True; mll = S["start"] + 100
            else: mll = peak - S["mll"]
        prof = bal - S["start"]
        if good >= 5 and prof > 0:
            amt = min(0.5 * prof, S["cap"])
            if cushion is not None: amt = min(amt, bal - (S["start"] + 100) - cushion)
            if amt >= 500:
                bal -= amt; paid += 0.9 * amt; npay += 1; good = 0
                if first is None: first = traded
                locked = True; mll = S["start"] + 100; peak = bal
    return paid, npay, False, first, traded

if __name__ == "__main__":
    D = os.environ.get("NQ_DATA", "data")
    days = pickle.load(open(os.path.join(D, "days.pkl"), "rb")); dates = [str(d["date"]) for d in days]
    res = pickle.load(open(os.path.join(D, "res_all.pkl"), "rb"))
    pts, low, n = lucid.day_series(res["noise"]["trades"], dates)
    starts = [k for k in range(0, len(pts) - 250, 3)]
    out = {}
    for size in ("50K", "100K", "150K"):
        S = SIZES[size]
        for m in (1, 2, 3, 4, 6, 8):
            if m > S["max_micro"]: continue
            for pol, cush in (("asap", None), ("cushion1500", 1500), ("cushion2500", 2500)):
                r = [funded_year(pts, low, n, s, m, S, cush) for s in starts]
                paid = np.array([x[0] for x in r]); br = np.array([x[2] for x in r]); npay = np.array([x[1] for x in r])
                firsts = [x[3] for x in r if x[3] is not None]
                hist = np.histogram(paid, bins=[0, 1, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 1e9])[0].tolist()
                out[f"{size}|{m}|{pol}"] = dict(size=size, micros=m, policy=pol, mean=round(float(paid.mean())), p25=round(float(np.percentile(paid, 25))),
                    p50=round(float(np.median(paid))), p75=round(float(np.percentile(paid, 75))), p90=round(float(np.percentile(paid, 90))),
                    zero=round(float((paid == 0).mean()), 3), breach=round(float(br.mean()), 3), npay=round(float(npay.mean()), 2),
                    first=int(np.median(firsts)) if firsts else None, hist=hist)
                if size == "50K": print(size, m, pol, {k: v for k, v in out[f"{size}|{m}|{pol}"].items() if k in ("mean", "p50", "p90", "zero", "breach", "npay", "first")})
    json.dump(out, open(os.path.join(D, "funded.json"), "w"))
