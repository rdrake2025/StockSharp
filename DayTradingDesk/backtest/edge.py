"""Edge tools data: pre-open day features + outcomes (for the similar-days outlook), intraday
'how much more can it move' tables, and adverse-excursion quantiles for stop placement.

Forecast skill is measured walk-forward: each day is predicted only from days BEFORE it.
"""
import os, pickle, json, math
import numpy as np

D = os.environ.get("NQ_DATA", "data")
days = pickle.load(open(os.path.join(D, "days.pkl"), "rb"))
res = pickle.load(open(os.path.join(D, "res_all.pkl"), "rb"))
dates = [str(d["date"]) for d in days]; di = {d: i for i, d in enumerate(dates)}
n = len(days)

# ---------------- features known at 9:30 ET
rth_ret = np.array([d["c"][-1] / d["o"][0] - 1 for d in days])
closes = np.array([d["c"][-1] for d in days])
F = np.full((n, 5), np.nan)   # gap%, overnight range%, prior-day range%, 20d vol (ann %), dist from 50d avg %
for i, d in enumerate(days):
    if i < 50 or d["roll"] or not np.isfinite(d["pdc"]) or not np.isfinite(d["onh"]): continue
    F[i, 0] = (d["o"][0] / d["pdc"] - 1) * 100
    F[i, 1] = (d["onh"] - d["onl"]) / d["pdc"] * 100
    F[i, 2] = (d["pdh"] - d["pdl"]) / d["pdc"] * 100
    F[i, 3] = rth_ret[i - 20:i].std() * math.sqrt(252) * 100
    F[i, 4] = (closes[i - 1] / closes[i - 50:i].mean() - 1) * 100
# ---------------- outcomes
rng_pts = np.array([d["h"].max() - d["l"].min() for d in days])
rng_pct = np.array([(d["h"].max() - d["l"].min()) / d["o"][0] * 100 for d in days])
oc_pct = rth_ret * 100
trend = np.array([abs(d["c"][-1] - d["o"][0]) / max(1e-9, d["h"].max() - d["l"].min()) > 0.6 for d in days])
def daily_pts(k):
    r = np.zeros(n)
    for t in res[k]["trades"]: r[di[t["date"]]] += t["pts"]
    return r
noise_pts = daily_pts("noise"); noise_ret = np.zeros(n)
for t in res["noise"]["trades"]: noise_ret[di[t["date"]]] += t["ret"] * 100

# ---------------- walk-forward skill of the kNN outlook
K = 60
def knn(i, pool):
    X = F[pool]; mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1
    z = (X - mu) / sd; zi = (F[i] - mu) / sd
    dist = np.sqrt(np.nansum((z - zi) ** 2, 1))
    return pool[np.argsort(dist)[:K]]
valid = np.where(~np.isnan(F).any(1))[0]
pred_rng, act_rng, pred_tr, act_tr, pred_noise, act_noise, naive_rng = [], [], [], [], [], [], []
for i in valid:
    pool = valid[valid < i - 1]
    if len(pool) < 250: continue
    nb = knn(i, pool)
    pred_rng.append(np.median(rng_pct[nb])); act_rng.append(rng_pct[i]); naive_rng.append(np.median(rng_pct[pool[-60:]]))
    pred_tr.append(trend[nb].mean()); act_tr.append(trend[i])
    pred_noise.append(noise_ret[nb].mean()); act_noise.append(noise_ret[i])
pred_rng, act_rng, naive_rng = map(np.array, (pred_rng, act_rng, naive_rng))
pred_tr, act_tr = np.array(pred_tr), np.array(act_tr); pred_noise, act_noise = np.array(pred_noise), np.array(act_noise)
mae_knn = np.mean(np.abs(np.log(pred_rng / act_rng))); mae_naive = np.mean(np.abs(np.log(naive_rng / act_rng)))
corr_rng = np.corrcoef(pred_rng, act_rng)[0, 1]
# trend-day calibration: bucket predicted probability into terciles
qs = np.quantile(pred_tr, [1/3, 2/3]); tb = []
for lo, hi, lab in [(-1, qs[0], "Low"), (qs[0], qs[1], "Mid"), (qs[1], 2, "High")]:
    m = (pred_tr > lo) & (pred_tr <= hi); tb.append(dict(label=lab, pred=round(float(pred_tr[m].mean()), 3), actual=round(float(act_tr[m].mean()), 3), n=int(m.sum())))
qn = np.quantile(pred_noise, [1/3, 2/3]); nbk = []
for lo, hi, lab in [(-1e9, qn[0], "Low"), (qn[0], qn[1], "Mid"), (qn[1], 1e9, "High")]:
    m = (pred_noise > lo) & (pred_noise <= hi); nbk.append(dict(label=lab, pred=round(float(pred_noise[m].mean()), 4), actual=round(float(act_noise[m].mean()), 4), n=int(m.sum())))
skill = dict(n=int(len(act_rng)), corr_range=round(float(corr_rng), 3), err_knn=round(float(mae_knn), 3), err_naive=round(float(mae_naive), 3),
             trend_buckets=tb, noise_buckets=nbk, base_trend=round(float(trend[valid].mean()), 3))
print("skill", json.dumps(skill))

# ---------------- intraday: how much more can it move?
checks = list(range(29, 360, 30))   # bar indices at 10:00 ... 15:30 closes
mv = []
for t in checks:
    hi_set, lo_set, ext = [], [], []
    for d in days[-750:]:          # last ~3 years
        hs, ls = d["h"][:t + 1].max(), d["l"][:t + 1].min(); H, L = d["h"].max(), d["l"].min()
        hi_set.append(H <= hs); lo_set.append(L >= ls)
        ext.append(((H - hs) + (ls - L)) / d["o"][0] * 100)     # extra range still to come, % of price
    ext = np.array(ext)
    mv.append(dict(t=570 + t + 1, p_hi=round(float(np.mean(hi_set)), 3), p_lo=round(float(np.mean(lo_set)), 3),
                   p_both=round(float(np.mean(np.array(hi_set) & np.array(lo_set))), 3),
                   ext=[round(float(x), 3) for x in np.percentile(ext, [50, 80, 95])]))
print("more", mv[0], mv[3], mv[-1])

# ---------------- stop placement: adverse excursion quantiles (% of entry price)
def mae_q(k):
    tr = res[k]["trades"]; w = [-t["mae"] / t["entry"] * 100 for t in tr if t["pts"] > 0]; l = [-t["mae"] / t["entry"] * 100 for t in tr if t["pts"] <= 0]
    return dict(win=[round(float(x), 4) for x in np.percentile(w, range(0, 101, 2))], loss=[round(float(x), 4) for x in np.percentile(l, range(0, 101, 2))], nw=len(w), nl=len(l))
mae = {k: mae_q(k) for k in ["noise", "drive", "lunch", "orb5_z", "gap_go"]}
print("noise winners MAE% p50/p80/p95:", mae["noise"]["win"][25], mae["noise"]["win"][40], mae["noise"]["win"][47])

# ---------------- per-day table for the browser kNN
rows = []
for i in valid:
    rows.append([dates[i]] + [round(float(x), 3) for x in F[i]] + [round(float(rng_pct[i]), 3), round(float(oc_pct[i]), 3), int(trend[i]), round(float(noise_ret[i]), 4), int(days[i]["h"].argmax() < 60 or days[i]["l"].argmin() < 60)])
out = dict(cols=["date", "gap", "onr", "pdr", "vol", "trend50", "rng", "oc", "trendday", "noise", "firsthour"], rows=rows, skill=skill, more=mv, mae=mae, K=K)
json.dump(out, open(os.path.join(D, "edge.json"), "w"))
print("rows", len(rows))
