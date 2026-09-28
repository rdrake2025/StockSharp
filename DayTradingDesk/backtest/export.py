"""Export backtest results + real NQ sessions to a compact JSON the web app embeds."""
import os, pickle, json, random, math
import numpy as np
import bt

D = os.environ["NQ_DATA"]
days = pickle.load(open(os.path.join(D, "days.pkl"), "rb"))
byd = {str(d["date"]): d for d in days}
res = pickle.load(open(os.path.join(D, "res_base.pkl"), "rb"))

def verdict(r):
    a, o = r["all"], r["oos"]; yrs = r["years"]; pos = sum(v["pts"] > 0 for v in yrs.values()) / len(yrs)
    if r["family"] == "Benchmark": return "bench"
    if a["pf"] >= 1.12 and o["pf"] >= 1.1 and pos >= 0.7 and a["sharpe"] > 0.3: return "edge"
    if a["pf"] >= 1.04 and o["pf"] >= 1.0: return "marginal"
    return "none"

def bars5(d):
    b = bt.five(d); base = d["o"][0]
    q = lambda a: [int(round((x - base) / bt.TICK)) for x in a]
    return dict(date=str(d["date"]), base=float(base), o=q(b["O"]), h=q(b["H"]), l=q(b["L"]), c=q(b["C"]), vw=q(b["VW"]),
                pdh=float(d["pdh"]) if np.isfinite(d["pdh"]) else None, pdl=float(d["pdl"]) if np.isfinite(d["pdl"]) else None,
                pdc=float(d["pdc"]) if np.isfinite(d["pdc"]) else None)

rng = random.Random(7)
out = {"meta": dict(first=str(days[0]["date"]), last=str(days[-1]["date"]), sessions=len(days), cost=bt.COST, split="2022-01-01"), "strats": []}
order = sorted(res, key=lambda k: (-res[k]["all"]["sharpe"]))
for k in order:
    r = res[k]; tr = r["trades"]
    # weekly equity (cumulative NQ points) for chart
    cum = 0; eq = []; last_week = None
    for t in sorted(tr, key=lambda t: (t["date"], t["i"])):
        cum += t["pts"]; wk = t["date"][:4] + "-" + str(np.datetime64(t["date"]).astype("datetime64[W]"))
        if wk != last_week: eq.append([t["date"], round(cum, 1)]); last_week = wk
        else: eq[-1] = [t["date"], round(cum, 1)]
    cp = np.cumsum([t["pts"] for t in sorted(tr, key=lambda t: (t["date"], t["i"]))])
    dd_pts = float(np.max(np.maximum.accumulate(np.concatenate([[0], cp])) - np.concatenate([[0], cp])))
    longs = [t for t in tr if t["side"] > 0]; shorts = [t for t in tr if t["side"] < 0]
    Rs = [t["R"] for t in tr if t["R"] is not None]
    hist = None
    if Rs:
        edges = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 5, 100]
        hist = [int(sum(1 for x in Rs if lo <= x < hi)) for lo, hi in zip([-100] + edges[:-1], edges)]
    # examples: a winner and a loser from 2023-2025 (realistic recent price levels)
    ex = []
    recent = [t for t in tr if t["date"] >= "2023-01-01" and t["i"] >= 5]
    for want in ("win", "loss"):
        pool = [t for t in recent if (t["pts"] > 0) == (want == "win")]
        if want == "win": pool = [t for t in pool if t["pts"] > 0.5 * np.median([p["pts"] for p in pool] or [0])]
        if pool:
            t = rng.choice(pool); d = byd[t["date"]]; same = [x for x in tr if x["date"] == t["date"]]
            ex.append(dict(kind=want, day=bars5(d), trades=[dict(side=x["side"], i=x["i"], j=x["j"], entry=x["entry"], exit=x["exit"],
                        stop=x["stop0"], pts=round(x["pts"], 2), why=x["why"]) for x in same]))
    out["strats"].append(dict(key=k, name=r["name"], family=r["family"], verdict=verdict(r),
        all={kk: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for kk, v in r["all"].items()},
        ins={kk: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for kk, v in r["ins"].items()},
        oos={kk: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for kk, v in r["oos"].items()},
        years={y: dict(n=v["n"], pts=round(v["pts"], 1), win=round(v["win"], 3)) for y, v in r["years"].items()},
        long=dict(n=len(longs), pts=round(sum(t["pts"] for t in longs), 1)), short=dict(n=len(shorts), pts=round(sum(t["pts"] for t in shorts), 1)),
        hist=hist, eq=eq, dd_pts=round(dd_pts, 1), examples=ex))

# time-of-day profile, 2023-2025: mean absolute 5-min move (pts) and share of volume
rec = [d for d in days if str(d["date"]) >= "2023-01-01"]
mv = np.mean([np.abs(bt.five(d)["C"] - bt.five(d)["O"]) for d in rec], axis=0)
rngp = np.mean([bt.five(d)["H"] - bt.five(d)["L"] for d in rec], axis=0)
vol = np.mean([d["v"].reshape(78, 5).sum(1) / max(d["v"].sum(), 1) for d in rec], axis=0)
out["tod"] = dict(range=[round(x, 2) for x in rngp], move=[round(x, 2) for x in mv], vol=[round(x * 100, 3) for x in vol])
# day stats
drng = [d["h"].max() - d["l"].min() for d in rec]; dpct = [(d["h"].max() - d["l"].min()) / d["o"][0] * 100 for d in rec]
oc = [abs(d["c"][-1] - d["o"][0]) / max(d["h"].max() - d["l"].min(), 1e-9) for d in rec]
out["daystats"] = dict(n=len(rec), med_range=round(float(np.median(drng)), 1), med_range_pct=round(float(np.median(dpct)), 2),
                       trend_share=round(float(np.mean(np.array(oc) > 0.6)), 3), chop_share=round(float(np.mean(np.array(oc) < 0.25)), 3),
                       hi_first_hour=round(float(np.mean([d["h"].argmax() < 60 or d["l"].argmin() < 60 for d in rec])), 3))
# noise-band sigma profile at each 30-min check (last 60 sessions), fraction
last = days[-60:]
sig = np.mean([np.abs(d["c"] / d["o"][0] - 1) for d in last], axis=0)
out["sigma"] = [round(float(sig[t]) * 100, 4) for t in range(29, 360, 30)]
# replay pool: 160 random real sessions across all years
pool = rng.sample(days[1:], 160)
out["replay"] = [bars5(d) for d in sorted(pool, key=lambda d: d["date"])]
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return float(o) if math.isfinite(o) else None
    if isinstance(o, np.bool_): return bool(o)
    return o
s = json.dumps(clean(out), separators=(",", ":"))
open(os.path.join(os.path.dirname(__file__), "..", "bt-data.json"), "w").write(s)
print("bytes", len(s)); print(json.dumps(out["daystats"]), out["sigma"][:4])
for st in out["strats"]: print(st["key"], st["verdict"], st["long"], st["short"])
