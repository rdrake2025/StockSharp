"""Regime analyst agent: WHEN does each strategy work?

Buckets every session by conditions known before the open (no look-ahead):
  * volatility: prior 20-session realized volatility of RTH returns, split into terciles (expanding, past-only cut-offs)
  * trend: prior close above/below its 50-session average
  * gap: size of today's opening gap vs prior close
  * weekday
and reports, per strategy, trades / avg points / profit factor in each bucket.
Also the Portfolio builder: daily P&L correlations and equal-weight combinations fixed in advance.
"""
import os, pickle, json, math
import numpy as np

D = os.environ.get("NQ_DATA", "data")
days = pickle.load(open(os.path.join(D, "days.pkl"), "rb"))
res = pickle.load(open(os.path.join(D, "res_all.pkl"), "rb"))
dates = [str(d["date"]) for d in days]; di = {d: i for i, d in enumerate(dates)}

closes = np.array([d["c"][-1] for d in days]); opens = np.array([d["o"][0] for d in days])
rth_ret = np.array([d["c"][-1] / d["o"][0] - 1 for d in days])
labels = {}
vol = np.full(len(days), np.nan)
for i in range(20, len(days)): vol[i] = rth_ret[i - 20:i].std() * math.sqrt(252)
for i, d in enumerate(days):
    lab = {}
    if i >= 120:
        past = vol[20:i]; t1, t2 = np.nanpercentile(past, [33.3, 66.7])
        lab["vol"] = "Low vol" if vol[i] < t1 else "Mid vol" if vol[i] < t2 else "High vol"
    if i >= 50:
        lab["trend"] = "Uptrend (above 50-day avg)" if closes[i - 1] > closes[i - 50:i].mean() else "Downtrend (below 50-day avg)"
    if np.isfinite(d["pdc"]) and not d["roll"]:
        g = abs(d["o"][0] / d["pdc"] - 1) * 100
        lab["gap"] = "Small gap (<0.3%)" if g < 0.3 else "Medium gap (0.3–1%)" if g < 1 else "Large gap (>1%)"
    lab["weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d["date"].weekday()]
    labels[dates[i]] = lab

DIMS = {"vol": ["Low vol", "Mid vol", "High vol"], "trend": ["Uptrend (above 50-day avg)", "Downtrend (below 50-day avg)"],
        "gap": ["Small gap (<0.3%)", "Medium gap (0.3–1%)", "Large gap (>1%)"], "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri"]}

def cell(tr):
    if not tr: return dict(n=0, avg=0.0, pf=None)
    p = np.array([t["pts"] for t in tr]); w = p[p > 0].sum(); l = -p[p <= 0].sum()
    return dict(n=len(p), avg=round(float(p.mean()), 2), pf=round(float(w / l), 2) if l > 0 else None)

KEYS = [k for k in res if k != "bench" and res[k]["all"]["pf"] > 1.05] + ["bench"]
regimes = {}
for k in KEYS:
    tr = res[k]["trades"]; regimes[k] = {}
    for dim, buckets in DIMS.items():
        regimes[k][dim] = {b: cell([t for t in tr if labels[t["date"]].get(dim) == b]) for b in buckets}

def daily_pts(k):
    r = np.zeros(len(dates))
    for t in res[k]["trades"]: r[di[t["date"]]] += t["pts"]
    return r

def stats(r):
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    cum = np.cumsum(r); dd = float(np.max(np.maximum.accumulate(cum) - cum))
    return dict(sharpe=round(float(sh), 2), total=round(float(r.sum()), 1), maxdd=round(dd, 1), ret_dd=round(float(r.sum() / dd), 2) if dd > 0 else None)

cand = ["noise", "drive", "lunch", "late_mom", "vwap_trend", "nr4_orb", "gap_go", "orb5_z"]
dp = {k: daily_pts(k) for k in cand}
corr = np.corrcoef([dp[k] for k in cand])
PORTS = {  # fixed in advance by evidence tier, not by searching combinations
    "Proven only": ["noise"],
    "Proven + promising": ["noise", "drive", "lunch"],
    "Momentum family": ["noise", "drive", "lunch", "late_mom", "vwap_trend"],
}
ports = {}
for name, ks in PORTS.items():
    r = sum(dp[k] for k in ks)
    s = stats(r); cum = np.cumsum(r)
    wk = []; last = None
    for i, d in enumerate(dates):
        w = str(np.datetime64(d).astype("datetime64[W]"))
        if w != last: wk.append([d, round(float(cum[i]), 1)]); last = w
        else: wk[-1] = [d, round(float(cum[i]), 1)]
    yrs = {}
    for i, d in enumerate(dates): yrs[d[:4]] = yrs.get(d[:4], 0) + r[i]
    ports[name] = dict(keys=ks, **s, eq=wk, up_years=sum(v > 0 for v in yrs.values()), years=len(yrs))
split = {}
for k in ["noise", "drive", "lunch", "late_mom", "gap_go", "orb5_z"]:
    split[k] = {}
    for half, (a, b) in {"2015–21": ("2015", "2022"), "2022–25": ("2022", "2026")}.items():
        for lab in DIMS["trend"]:
            tr = [t for t in res[k]["trades"] if a <= t["date"] < b and labels[t["date"]].get("trend") == lab]
            split[k][f"{half}|{lab.split()[0]}"] = cell(tr)
out = dict(trend_split=split, regimes=regimes, dims=DIMS, corr=dict(keys=cand, m=corr.round(2).tolist()), portfolios=ports)
json.dump(out, open(os.path.join(D, "regime.json"), "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
for k in ["noise", "drive", "lunch", "late_mom", "gap_go"]:
    print(k, {dim: {b: (c["n"], c["avg"], c["pf"]) for b, c in v.items()} for dim, v in regimes[k].items() if dim in ("vol", "trend")})
print(np.array(corr).round(2))
for n, p in ports.items(): print(n, {k: v for k, v in p.items() if k != "eq"})
