"""Statistical evidence for each strategy.

1. t-test on daily returns, with Holm-Bonferroni correction across every strategy tested.
2. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): probability the true Sharpe > 0 after
   accounting for the number of strategies/variants tried, skew and fat tails.
3. Block bootstrap (20-day blocks) 95% confidence intervals for annual Sharpe and profit factor.
4. Random-direction test: same trades, same timing, coin-flip direction (does the signal pick the side?).
5. Random-day test: same side and intraday timing, placed on random other sessions (does the day selection matter?).
6. Walk-forward: parameters (noise band) and strategy selection chosen only from past data, traded the next year.
"""
import os, pickle, math, json, random
import numpy as np
from statistics import NormalDist
import bt
from strategies_more import MORE

N01 = NormalDist()
D = os.environ.get("NQ_DATA", "data")
days = pickle.load(open(os.path.join(D, "days.pkl"), "rb"))
res = pickle.load(open(os.path.join(D, "res_all.pkl"), "rb"))
dates = [str(d["date"]) for d in days]; di = {d: i for i, d in enumerate(dates)}
N_TRIALS = len(res) + 9 + 4 + 4          # strategies + robustness variants run earlier
rng = np.random.default_rng(42)

def daily(tr):
    r = np.zeros(len(dates))
    for t in tr: r[di[t["date"]]] += t["ret"]
    return r

def sharpe(r): return r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0.0

def dsr(r, sr_trials):
    """Deflated Sharpe ratio (per-period Sharpe)."""
    T = len(r); sr = r.mean() / r.std()
    sk = ((r - r.mean()) ** 3).mean() / r.std() ** 3; ku = ((r - r.mean()) ** 4).mean() / r.std() ** 4
    v = np.var(sr_trials); g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * N01.inv_cdf(1 - 1 / N_TRIALS) + g * N01.inv_cdf(1 - 1 / (N_TRIALS * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr * sr))
    return N01.cdf(z), sr0 * math.sqrt(252)

def block_boot(r, pts_by_day, reps=2000, block=20):
    n = len(r); nb = n // block + 1; shs, pfs = [], []
    for _ in range(reps):
        starts = rng.integers(0, n - block, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        x = r[idx]; shs.append(sharpe(x))
        p = pts_by_day[idx]; w = p[p > 0].sum(); l = -p[p < 0].sum(); pfs.append(w / l if l > 0 else np.nan)
    return np.nanpercentile(shs, [2.5, 97.5]).tolist(), np.nanpercentile(pfs, [2.5, 97.5]).tolist()

def rand_dir(tr, reps=2000):
    g = np.array([t["pts"] + bt.COST for t in tr])   # gross pts in the traded direction
    actual = (g - bt.COST).sum()
    sims = np.array([(g * rng.choice([-1, 1], len(g)) - bt.COST).sum() for _ in range(reps)])
    return float((sims >= actual).mean()), sims

def rand_day(tr, reps=500):
    """Same side, entry bar and exit bar, but on a random session (market in, market out, no stops)."""
    O = np.array([d["o"] for d in days]); C = np.array([d["c"] for d in days])
    ii = np.array([t["i"] for t in tr]); jj = np.array([t["j"] for t in tr]); ss = np.array([t["side"] for t in tr])
    own = ((C[[di[t["date"]] for t in tr], jj] - O[[di[t["date"]] for t in tr], ii]) * ss - bt.COST).sum()
    sims = []
    for _ in range(reps):
        k = rng.integers(0, len(days), len(tr))
        sims.append(((C[k, jj] - O[k, ii]) * ss - bt.COST).sum())
    sims = np.array(sims)
    return float((sims >= own).mean()), float(own)

def walk_forward_noise():
    grid = [(lb, vm) for lb in (10, 14, 20) for vm in (0.8, 1.0, 1.2)]
    runs = {g: bt.run(days, bt.noise_boundary(*g)) for g in grid}
    by_year = {g: {} for g in grid}
    for g, tr in runs.items():
        for t in tr: by_year[g].setdefault(t["date"][:4], []).append(t)
    out = []; years = sorted({t["date"][:4] for t in runs[grid[0]]})
    for y in years:
        prior = [str(int(y) - k) for k in (1, 2, 3)]
        if not all(p in by_year[grid[0]] for p in prior): continue
        def score(g):
            r = daily([t for p in prior for t in by_year[g].get(p, [])]); return sharpe(r)
        best = max(grid, key=score)
        tr = by_year[best].get(y, [])
        out.append(dict(year=y, params=f"lookback {best[0]}, band ×{best[1]}", pts=round(sum(t["pts"] for t in tr), 1), n=len(tr)))
    return out

def walk_forward_select(topk=3):
    """Each year, pick the top-k strategies by prior-3-year Sharpe from the WHOLE universe, trade them equally next year."""
    keys = [k for k in res if k != "bench"]
    yr_ret = {k: {} for k in keys}; yr_pts = {k: {} for k in keys}
    for k in keys:
        for t in res[k]["trades"]:
            y = t["date"][:4]; yr_pts[k][y] = yr_pts[k].get(y, 0) + t["pts"]
    dr = {k: daily(res[k]["trades"]) for k in keys}
    yidx = {}
    for i, d in enumerate(dates): yidx.setdefault(d[:4], []).append(i)
    out = []
    for y in sorted(yidx):
        prior = [str(int(y) - k) for k in (1, 2, 3)]
        if not all(p in yidx for p in prior): continue
        pi = sum((yidx[p] for p in prior), [])
        ranked = sorted(keys, key=lambda k: -sharpe(dr[k][pi]))[:topk]
        pts = sum(yr_pts[k].get(y, 0) for k in ranked) / topk
        out.append(dict(year=y, picks=[res[k]["name"] for k in ranked], keys=ranked, pts=round(pts, 1)))
    return out

def main():
    keys = list(res)
    dr = {k: daily(res[k]["trades"]) for k in keys}
    sr_trials = [dr[k].mean() / dr[k].std() for k in keys if dr[k].std() > 0]
    rows = {}
    for k in keys:
        r = dr[k]; n = len(r); t = r.mean() / (r.std(ddof=1) / math.sqrt(n)); p = 1 - N01.cdf(t)
        pts_day = np.zeros(len(dates))
        for tt in res[k]["trades"]: pts_day[di[tt["date"]]] += tt["pts"]
        d_, sr0 = dsr(r, sr_trials)
        ci_sh, ci_pf = block_boot(r, pts_day)
        rows[k] = dict(t=round(t, 2), p=p, dsr=round(d_, 3), sr0=round(sr0, 2), ci_sharpe=[round(x, 2) for x in ci_sh], ci_pf=[round(x, 2) for x in ci_pf])
    # Holm-Bonferroni across all strategies (one-sided, alpha 5%)
    order = sorted(keys, key=lambda k: rows[k]["p"]); m = len(order); stop = False
    for i, k in enumerate(order):
        thr = 0.05 / (m - i)
        rows[k]["holm"] = (not stop) and rows[k]["p"] <= thr
        if not rows[k]["holm"]: stop = True
        rows[k]["p_adj"] = min(1.0, max(rows[o]["p"] * (m - j) for j, o in enumerate(order[:i + 1])))
    # randomization tests for the strategies that look positive
    for k in keys:
        if res[k]["all"]["pf"] > 1.08 and k != "bench":
            p_dir, sims = rand_dir(res[k]["trades"]); p_day, _ = rand_day(res[k]["trades"])
            rows[k].update(p_dir=p_dir, p_day=p_day, dir_hist=np.histogram(sims, bins=30)[0].tolist(),
                           dir_edges=[round(float(x)) for x in np.histogram(sims, bins=30)[1]], actual=round(res[k]["all"]["tot_pts"]))
    wf_noise = walk_forward_noise(); wf_sel = walk_forward_select(3)
    # correlation of daily P&L among the strongest strategies + equal-weight portfolio
    strong = [k for k in keys if rows[k]["holm"] and k != "bench"]
    corr = np.corrcoef([dr[k] for k in strong]).round(2).tolist() if len(strong) > 1 else []
    port = sum(dr[k] for k in strong) / max(1, len(strong)) if strong else np.zeros(len(dates))
    out = dict(n_trials=N_TRIALS, rows=rows, wf_noise=wf_noise, wf_select=wf_sel, strong=strong, corr=corr,
               portfolio=dict(sharpe=round(sharpe(port), 2), keys=strong))
    json.dump(out, open(os.path.join(D, "proof.json"), "w"), default=float)
    print(f"trials={N_TRIALS}")
    for k in order:
        r = rows[k]
        print(f"{k:11s} t={r['t']:5.2f} p={r['p']:.5f} p_holm={r['p_adj']:.4f} {'PASS' if r['holm'] else '    '} DSR={r['dsr']:.3f} "
              f"Sh95%={r['ci_sharpe']} PF95%={r['ci_pf']} " + (f"p_dir={r['p_dir']:.4f} p_day={r['p_day']:.4f}" if 'p_dir' in r else ""))
    print("WF noise:", [(w['year'], w['pts']) for w in wf_noise], "total", round(sum(w['pts'] for w in wf_noise)))
    print("WF select top3:", [(w['year'], w['pts'], w['keys']) for w in wf_sel], "total", round(sum(w['pts'] for w in wf_sel)))
    print("strong:", strong, "portfolio sharpe", out["portfolio"]["sharpe"]); print(np.array(corr))

if __name__ == "__main__":
    main()
