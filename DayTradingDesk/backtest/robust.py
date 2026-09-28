"""Parameter and cost sensitivity for the strategies that survived the base test."""
import os, pickle, importlib
import numpy as np
import bt

days = pickle.load(open(os.path.join(os.environ["NQ_DATA"], "days.pkl"), "rb"))
def row(label, fn):
    tr = bt.run(days, fn); m = bt.metrics(tr, len(days))
    o = bt.metrics([t for t in tr if t["date"] >= "2022"], sum(str(d["date"]) >= "2022" for d in days))
    yrs = {}
    for t in tr: yrs[t["date"][:4]] = yrs.get(t["date"][:4], 0) + t["pts"]
    print(f"{label:34s} n={m['n']:5d} win={m['win']:.0%} PF={m['pf']:.2f} avg={m['avg_pts']:6.2f} Sh={m['sharpe']:5.2f} OOS_Sh={o['sharpe']:5.2f} +yrs={sum(v>0 for v in yrs.values())}/{len(yrs)}")
    return m

print("--- noise boundary: lookback x band multiplier")
for lb in (10, 14, 20):
    for vm in (0.8, 1.0, 1.2):
        row(f"noise lb={lb} vm={vm}", bt.noise_boundary(lb, vm))
print("--- gap and go: min gap")
for g in (0.3, 0.5, 0.8, 1.0):
    row(f"gap_go min={g}%", bt.gap_go(g))
print("--- 5-min ORB target")
for t in (2, 5, 10, 1000):
    row(f"orb5 tgt={t}R", bt.orb5_zarattini(t))
print("--- cost sensitivity (NQ points per round trip)")
for c in (0.5, 0.75, 1.12, 1.5, 2.0):
    bt.COST = c
    row(f"noise   cost={c}", bt.noise_boundary(14, 1.0))
    row(f"gap_go  cost={c}", bt.gap_go(0.5))
    row(f"orb5_z  cost={c}", bt.orb5_zarattini(10))
    row(f"orb30_2r cost={c}", bt.orb(30, 2.0))
