"""Per-session features for the in-app Market Lab: every RTH session 2015-2025, compact column arrays."""
import os, pickle, json, datetime as dt
import numpy as np
import bt, window
from strategies_scout import FOMC
D = os.environ.get("NQ_DATA", "data")
days = pickle.load(open(os.path.join(D, "days.pkl"), "rb"))
res = pickle.load(open(os.path.join(D, "res_all.pkl"), "rb"))
full = {}; 
for t in res["noise"]["trades"]: full[t["date"]] = full.get(t["date"], 0) + t["pts"]
pers = {}
for t in bt.run(days, window.noise_window_hold(119, 150)): pers[t["date"]] = pers.get(t["date"], 0) + t["pts"]
def opex(d):  # third Friday
    return d.weekday() == 4 and 15 <= d.day <= 21
cols = {k: [] for k in "d wd gap onr or30 or30r rng oc path hi lo trend fill fillm ibu ibd vol pday ev nz ps".split()}
rngs = []
for i, x in enumerate(days):
    o, h, l, c = x["o"], x["h"], x["l"], x["c"]; o0 = o[0]; date = x["date"]; ds = str(date)
    pdc = x["pdc"]; ok = np.isfinite(pdc) and not x["roll"]
    r = (h.max() - l.min()) / o0 * 100
    prior = rngs[-20:]; vol = None
    if len(prior) >= 20:
        q = np.percentile(rngs[-250:], [33, 67]) if len(rngs) >= 60 else None
        m = np.mean(prior); vol = 0 if q is not None and m < q[0] else 2 if q is not None and m > q[1] else 1
    rngs.append(r)
    gap = (o0 / pdc - 1) * 100 if ok else None
    fill = fm = None
    if gap is not None and abs(gap) >= .1:
        idx = np.where(l <= pdc)[0] if gap > 0 else np.where(h >= pdc)[0]
        fill = int(len(idx) > 0); fm = int(idx[0]) if len(idx) else None
    oc = (c[-1] / o0 - 1) * 100
    ib_h, ib_l = h[:60].max(), l[:60].min()
    cols["d"].append(ds); cols["wd"].append(date.weekday()); cols["gap"].append(None if gap is None else round(gap, 3))
    cols["onr"].append(round((x["onh"] - x["onl"]) / o0 * 100, 3) if np.isfinite(x["onh"]) and ok else None)
    cols["or30"].append(round((c[29] / o0 - 1) * 100, 3)); cols["or30r"].append(round((h[:30].max() - l[:30].min()) / o0 * 100, 3))
    cols["rng"].append(round(r, 3)); cols["oc"].append(round(oc, 3))
    cols["path"].append([int(round((c[m] / o0 - 1) * 1e4)) for m in range(29, 390, 30)])
    cols["hi"].append(int(np.argmax(h))); cols["lo"].append(int(np.argmin(l)))
    cols["trend"].append(int(abs(oc) / r > .6) * (1 if oc > 0 else -1) if r > 0 else 0)
    cols["fill"].append(fill); cols["fillm"].append(fm)
    cols["ibu"].append(int(h[60:].max() > ib_h)); cols["ibd"].append(int(l[60:].min() < ib_l))
    cols["vol"].append(vol); cols["pday"].append(None if i == 0 else (1 if days[i-1]["c"][-1] > days[i-1]["o"][0] else -1))
    cols["ev"].append(("F" if ds in FOMC else "") + ("O" if opex(date) else ""))
    cols["nz"].append(round(full.get(ds, 0), 2)); cols["ps"].append(round(pers.get(ds, 0), 2))
json.dump(cols, open(os.path.join(D, "study.json"), "w"), separators=(",", ":"))
n = len(cols["d"]); print(n, "sessions", os.path.getsize(os.path.join(D, "study.json")) // 1024, "KB")
g = [(gp, f) for gp, f in zip(cols["gap"], cols["fill"]) if f is not None]
print("gap fill rate", np.mean([f for _, f in g]), "trend share", np.mean([t != 0 for t in cols["trend"]]), "up days", np.mean([x > 0 for x in cols["oc"]]))
