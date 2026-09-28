"""Intraday strategy backtests on NQ futures, 1-minute bars, RTH only (9:30-16:00 ET).

Conventions (deliberately conservative):
  * Signals are evaluated on completed bars; market entries fill at the NEXT 1-minute bar open.
  * Stop-entries fill at the trigger price, or at the bar open if price gapped through it.
  * Stops and targets are checked on 1-minute bars. If a bar touches both, the stop wins.
  * Profit targets need price to trade one tick THROUGH the level to count as filled.
  * Every open position is flattened at the 15:59 close. One position at a time.
  * Round-trip cost (commission + fees + slippage) is subtracted from every trade: COST points.
"""
import os, pickle, json, math
import numpy as np

TICK = 0.25
COST = float(os.environ.get("COST", 0.75))   # NQ points per round trip (~$15 per NQ / $1.50 + fees per MNQ)
EOD = 389

# ---------------------------------------------------------------- helpers
def five(d):
    """Aggregate 1-min RTH arrays to 5-minute bars, with VWAP at each 5-min close."""
    o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
    tp = (h + l + c) / 3
    cv = np.cumsum(v); cpv = np.cumsum(tp * v)
    vw1 = np.where(cv > 0, cpv / np.maximum(cv, 1e-9), np.cumsum(tp) / np.arange(1, 391))
    # volume-weighted standard deviation of price around VWAP
    cpv2 = np.cumsum(tp * tp * v)
    var = np.where(cv > 0, cpv2 / np.maximum(cv, 1e-9) - vw1 ** 2, 0)
    sd1 = np.sqrt(np.maximum(var, 0))
    O = o[::5]; C = c[4::5]
    H = h.reshape(78, 5).max(1); L = l.reshape(78, 5).min(1)
    return dict(O=O, H=H, L=L, C=C, VW=vw1[4::5], SD=sd1[4::5], vw1=vw1)

def ema(x, n):
    k = 2 / (n + 1); out = np.empty_like(x); e = x[0]
    for i, xi in enumerate(x):
        e = xi * k + e * (1 - k); out[i] = e
    return out

def trade(d, i, side, entry, stop, target=None, end=EOD, stop_entry=False, trail=None):
    """Simulate one trade on 1-min bars starting at bar i. Returns dict or None."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    if stop_entry:  # fill at trigger, or worse if the bar opened beyond it
        entry = max(entry, o[i]) if side > 0 else min(entry, o[i])
    risk = (entry - stop) * side
    if risk <= 0:
        return None
    exit_px, why, j = None, "eod", end
    for j in range(i, end + 1):
        if trail is not None:
            s2 = trail(j)
            if s2 is not None:
                stop = max(stop, s2) if side > 0 else min(stop, s2)
        if side > 0:
            if l[j] <= stop:
                exit_px = min(stop, o[j]) if j > i else stop; why = "stop"; break
            if target is not None and h[j] >= target + TICK:
                exit_px = max(target, o[j]) if j > i else target; why = "target"; break
        else:
            if h[j] >= stop:
                exit_px = max(stop, o[j]) if j > i else stop; why = "stop"; break
            if target is not None and l[j] <= target - TICK:
                exit_px = min(target, o[j]) if j > i else target; why = "target"; break
    if exit_px is None:
        exit_px = c[end]; j = end
    pts = (exit_px - entry) * side - COST
    return dict(date=str(d["date"]), side=side, i=i, j=j, entry=float(entry), exit=float(exit_px),
                stop0=float(entry - risk * side), risk=float(risk), pts=float(pts), R=float(pts / risk),
                ret=float(pts / entry), why=why)

def first_cross_up(arr, level, start):
    idx = np.nonzero(arr[start:] > level)[0]
    return start + idx[0] if len(idx) else None

# ---------------------------------------------------------------- strategies
def orb(minutes, tgt_r):
    """Opening range breakout: stop-entry on first break of the N-minute range, stop at the other side."""
    def f(d, ctx):
        n = minutes
        hi, lo = d["h"][:n].max(), d["l"][:n].min()
        if hi - lo < 4 * TICK: return None
        iu = first_cross_up(d["h"], hi, n); idn = first_cross_up(-d["l"], -lo, n)
        if iu is None and idn is None: return None
        if iu is not None and (idn is None or iu < idn):
            if iu > 300: return None
            tg = hi + tgt_r * (hi - lo) if tgt_r else None
            return trade(d, iu, 1, hi + TICK, lo, tg, stop_entry=True)
        if idn > 300: return None
        tg = lo - tgt_r * (hi - lo) if tgt_r else None
        return trade(d, idn, -1, lo - TICK, hi, tg, stop_entry=True)
    return f

def orb5_zarattini(tgt_r=10):
    """Zarattini & Aziz (2023): trade the direction of the first 5-min candle from 9:35, stop at its extreme, 10R or EOD."""
    def f(d, ctx):
        o5, c5 = d["o"][0], d["c"][4]
        hi, lo = d["h"][:5].max(), d["l"][:5].min()
        if c5 == o5 or hi - lo < 2 * TICK: return None
        side = 1 if c5 > o5 else -1
        e = d["o"][5]; st = lo if side > 0 else hi
        r = (e - st) * side
        if r <= 0: return None
        return trade(d, 5, side, e, st, e + side * tgt_r * r)
    return f

def intraday_momentum(sig_end=30):
    """Gao, Han, Li & Zhou (2018): first half-hour return (from prior close) predicts the last half hour."""
    def f(d, ctx):
        if not np.isfinite(d["pdc"]) or d["roll"]: return None
        r1 = d["c"][sig_end - 1] / d["pdc"] - 1
        if r1 == 0: return None
        side = 1 if r1 > 0 else -1
        e = d["o"][360]
        return trade(d, 360, side, e, e - side * 1e6)
    return f

def noise_boundary(lookback=14, vm=1.0):
    """Zarattini, Aziz & Barbon (2024) 'Beat the Market': trade breakouts of a time-of-day noise band,
    checked every 30 min, trailing stop at the tighter of band and VWAP."""
    def f(d, ctx):
        hist = ctx["hist"]
        if len(hist) < lookback or not np.isfinite(d["pdc"]) or d["roll"]: return None
        sig = np.mean([np.abs(x["c"] / x["o"][0] - 1) for x in hist[-lookback:]], axis=0) * vm
        op = d["o"][0]
        ub = max(op, d["pdc"]) * (1 + sig); lb = min(op, d["pdc"]) * (1 - sig)
        vw = five(d)["vw1"]; c = d["c"]
        trades = []; pos = 0; entry = i_in = None
        checks = list(range(29, 360, 30))  # bar closes at 10:00, 10:30, ... 15:30
        for t in checks:
            if pos == 0:
                if c[t] > ub[t]: pos, entry, i_in = 1, d["o"][t + 1], t + 1
                elif c[t] < lb[t]: pos, entry, i_in = -1, d["o"][t + 1], t + 1
            else:
                stop = max(ub[t], vw[t]) if pos > 0 else min(lb[t], vw[t])
                if (pos > 0 and c[t] < stop) or (pos < 0 and c[t] > stop):
                    ex = d["o"][t + 1]
                    trades.append(_mk(d, pos, i_in, t + 1, entry, ex, "trail")); pos = 0
                    if c[t] > ub[t]: pos, entry, i_in = 1, d["o"][t + 1], t + 1
                    elif c[t] < lb[t]: pos, entry, i_in = -1, d["o"][t + 1], t + 1
        if pos: trades.append(_mk(d, pos, i_in, EOD, entry, c[EOD], "eod"))
        return trades or None
    return f

def _mk(d, side, i, j, entry, ex, why):
    pts = (ex - entry) * side - COST
    return dict(date=str(d["date"]), side=side, i=int(i), j=int(j), entry=float(entry), exit=float(ex), stop0=None,
                risk=None, pts=float(pts), R=None, ret=float(pts / entry), why=why)

def vwap_pullback(tgt_r=2.0):
    """Trend pullback: 30+ min on one side of VWAP, pullback bar tags VWAP and closes back on the trend side."""
    def f(d, ctx):
        b = five(d); O, H, L, C, VW = b["O"], b["H"], b["L"], b["C"], b["VW"]
        for k in range(7, 66):
            above = np.all(C[k - 6:k] > VW[k - 6:k]); below = np.all(C[k - 6:k] < VW[k - 6:k])
            if above and L[k] <= VW[k] and C[k] > VW[k]:
                st = min(L[k - 2:k + 1].min(), VW[k]) - TICK
                e = d["o"][5 * (k + 1)]
                if e - st < 4: st = e - 4
                return trade(d, 5 * (k + 1), 1, e, st, e + tgt_r * (e - st) if tgt_r else None)
            if below and H[k] >= VW[k] and C[k] < VW[k]:
                st = max(H[k - 2:k + 1].max(), VW[k]) + TICK
                e = d["o"][5 * (k + 1)]
                if st - e < 4: st = e + 4
                return trade(d, 5 * (k + 1), -1, e, st, e - tgt_r * (st - e) if tgt_r else None)
        return None
    return f

def vwap_reversion(k_sd=2.0):
    """Fade a stretch beyond VWAP +/- k standard deviations once a 5-min bar closes back inside; target VWAP."""
    def f(d, ctx):
        b = five(d); H, L, C, VW, SD = b["H"], b["L"], b["C"], b["VW"], b["SD"]
        for k in range(6, 66):
            up, dn = VW[k] + k_sd * SD[k], VW[k] - k_sd * SD[k]
            if H[k - 1] > VW[k - 1] + k_sd * SD[k - 1] and C[k] < up and C[k] > VW[k]:
                e = d["o"][5 * (k + 1)]; st = max(H[k - 1], H[k]) + TICK
                if e >= VW[k] + 2: return trade(d, 5 * (k + 1), -1, e, st, VW[k])
            if L[k - 1] < VW[k - 1] - k_sd * SD[k - 1] and C[k] > dn and C[k] < VW[k]:
                e = d["o"][5 * (k + 1)]; st = min(L[k - 1], L[k]) - TICK
                if e <= VW[k] - 2: return trade(d, 5 * (k + 1), 1, e, st, VW[k])
        return None
    return f

def ema_pullback(tgt_r=2.0):
    """5-min 9/20 EMA trend pullback: EMAs stacked and sloping, bar tags the 20 EMA and closes back with the trend."""
    def f(d, ctx):
        b = five(d); H, L, C = b["H"], b["L"], b["C"]
        e9, e20 = ema(C, 9), ema(C, 20)
        for k in range(8, 66):
            if e9[k] > e20[k] and e20[k] > e20[k - 3] and L[k] <= e20[k] and C[k] > e20[k]:
                e = d["o"][5 * (k + 1)]; st = min(L[k], L[k - 1]) - TICK
                if e - st < 4: st = e - 4
                return trade(d, 5 * (k + 1), 1, e, st, e + tgt_r * (e - st))
            if e9[k] < e20[k] and e20[k] < e20[k - 3] and H[k] >= e20[k] and C[k] < e20[k]:
                e = d["o"][5 * (k + 1)]; st = max(H[k], H[k - 1]) + TICK
                if st - e < 4: st = e + 4
                return trade(d, 5 * (k + 1), -1, e, st, e - tgt_r * (st - e))
        return None
    return f

def pdh_fail(tgt_r=2.0):
    """Failed breakout of prior-day high/low: poke through, then a 5-min close back inside -> fade."""
    def f(d, ctx):
        if d["roll"] or not np.isfinite(d["pdh"]): return None
        b = five(d); H, L, C = b["H"], b["L"], b["C"]
        pdh, pdl = d["pdh"], d["pdl"]
        if d["o"][0] > pdh or d["o"][0] < pdl: return None     # opened outside: not a test of the level
        broke_u = broke_d = False; hx, lx = -1e9, 1e9
        for k in range(0, 30):  # until 12:00
            if H[k] > pdh + 2 * TICK: broke_u = True
            if L[k] < pdl - 2 * TICK: broke_d = True
            hx = max(hx, H[k]); lx = min(lx, L[k])
            if broke_u and C[k] < pdh:
                e = d["o"][5 * (k + 1)]; st = hx + TICK
                return trade(d, 5 * (k + 1), -1, e, st, e - tgt_r * (st - e))
            if broke_d and C[k] > pdl:
                e = d["o"][5 * (k + 1)]; st = lx - TICK
                return trade(d, 5 * (k + 1), 1, e, st, e + tgt_r * (e - st))
        return None
    return f

def pdh_break(tgt_r=2.0):
    """Breakout continuation: 5-min close beyond prior-day high/low, stop back inside at the breakout bar's far side."""
    def f(d, ctx):
        if d["roll"] or not np.isfinite(d["pdh"]): return None
        b = five(d); H, L, C = b["H"], b["L"], b["C"]
        pdh, pdl = d["pdh"], d["pdl"]
        if d["o"][0] > pdh or d["o"][0] < pdl: return None
        for k in range(0, 60):
            if C[k] > pdh:
                e = d["o"][5 * (k + 1)]; st = min(L[k], pdh) - TICK
                if e - st < 4: st = e - 4
                return trade(d, 5 * (k + 1), 1, e, st, e + tgt_r * (e - st) if tgt_r else None)
            if C[k] < pdl:
                e = d["o"][5 * (k + 1)]; st = max(H[k], pdl) + TICK
                if st - e < 4: st = e + 4
                return trade(d, 5 * (k + 1), -1, e, st, e - tgt_r * (st - e) if tgt_r else None)
        return None
    return f

def gap_fill(lo_pct=0.3, hi_pct=1.0, mode="open"):
    """Fade a moderate opening gap toward the prior RTH close."""
    def f(d, ctx):
        if d["roll"] or not np.isfinite(d["pdc"]): return None
        g = (d["o"][0] / d["pdc"] - 1) * 100
        if not (lo_pct <= abs(g) <= hi_pct): return None
        side = -1 if g > 0 else 1
        if mode == "open":
            e = d["o"][0]; gap = abs(e - d["pdc"])
            return trade(d, 0, side, e, e - side * gap, d["pdc"], end=119)   # fill by 11:30 or out
        hi, lo = d["h"][:15].max(), d["l"][:15].min()
        if side < 0:
            i = first_cross_up(-d["l"], -lo, 15)
            if i is None or i > 120 or lo <= d["pdc"]: return None
            return trade(d, i, -1, lo - TICK, hi, d["pdc"], stop_entry=True)
        i = first_cross_up(d["h"], hi, 15)
        if i is None or i > 120 or hi >= d["pdc"]: return None
        return trade(d, i, 1, hi + TICK, lo, d["pdc"], stop_entry=True)
    return f

def gap_go(min_pct=0.5):
    """Gap-and-go: gap >= min_pct, trade the 15-min range break in the gap's direction, hold to close."""
    def f(d, ctx):
        if d["roll"] or not np.isfinite(d["pdc"]): return None
        g = (d["o"][0] / d["pdc"] - 1) * 100
        if abs(g) < min_pct: return None
        hi, lo = d["h"][:15].max(), d["l"][:15].min()
        if g > 0:
            i = first_cross_up(d["h"], hi, 15)
            if i is None or i > 180: return None
            return trade(d, i, 1, hi + TICK, lo, None, stop_entry=True)
        i = first_cross_up(-d["l"], -lo, 15)
        if i is None or i > 180: return None
        return trade(d, i, -1, lo - TICK, hi, None, stop_entry=True)
    return f

def bench_long(d, ctx):
    return trade(d, 0, 1, d["o"][0], d["o"][0] - 1e6)

# ---------------------------------------------------------------- runner & metrics
STRATS = {
    # key: (name, family, fn, playbook-ready description)
    "orb5_z":   ("5-min ORB, first-candle direction", "Breakout", orb5_zarattini(10)),
    "orb15_2r": ("15-min ORB, 2R target", "Breakout", orb(15, 2.0)),
    "orb15_eod":("15-min ORB, hold to close", "Breakout", orb(15, None)),
    "orb30_2r": ("30-min ORB, 2R target", "Breakout", orb(30, 2.0)),
    "orb30_1r": ("30-min ORB, 1R target", "Breakout", orb(30, 1.0)),
    "noise":    ("Noise-boundary momentum", "Momentum", noise_boundary(14, 1.0)),
    "imom":     ("Intraday momentum (last 30 min)", "Momentum", intraday_momentum(30)),
    "vwap_pb":  ("VWAP pullback, 2R", "Trend", vwap_pullback(2.0)),
    "ema_pb":   ("9/20 EMA pullback, 2R", "Trend", ema_pullback(2.0)),
    "vwap_rev": ("VWAP 2σ mean reversion", "Reversion", vwap_reversion(2.0)),
    "pdh_fail": ("Failed prior-day high/low", "Reversion", pdh_fail(2.0)),
    "pdh_break":("Prior-day high/low breakout", "Breakout", pdh_break(2.0)),
    "gap_fill": ("Gap fill, fade at open", "Reversion", gap_fill(0.3, 1.0, "open")),
    "gap_fill15":("Gap fill, 15-min break entry", "Reversion", gap_fill(0.3, 1.0, "break")),
    "gap_go":   ("Gap and go", "Breakout", gap_go(0.5)),
    "bench":    ("Benchmark: long every open→close", "Benchmark", bench_long),
}

def run(days, fn):
    out = []; hist = []
    for d in days:
        r = fn(d, {"hist": hist})
        if r:
            out.extend(r if isinstance(r, list) else [r])
        hist.append(d)
        if len(hist) > 40: hist.pop(0)
    return out

def metrics(tr, days_n):
    if not tr: return {}
    p = np.array([t["pts"] for t in tr]); r = np.array([t["ret"] for t in tr])
    w = p[p > 0]; ls = p[p <= 0]
    Rs = np.array([t["R"] for t in tr if t["R"] is not None])
    # daily returns (sum per date), zero on days without trades
    by = {}
    for t in tr: by[t["date"]] = by.get(t["date"], 0) + t["ret"]
    dr = np.array(list(by.values()) + [0.0] * max(0, days_n - len(by)))
    eq = np.cumsum(sorted_daily(tr))
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0
    return dict(n=len(tr), win=float(len(w) / len(p)), avg_pts=float(p.mean()), tot_pts=float(p.sum()),
                pf=float(w.sum() / -ls.sum()) if ls.sum() < 0 else float("inf"),
                avg_R=float(Rs.mean()) if len(Rs) else None,
                sharpe=float(dr.mean() / dr.std() * math.sqrt(252)) if dr.std() > 0 else 0.0,
                tot_ret=float(r.sum()), maxdd=dd, avg_win=float(w.mean()) if len(w) else 0, avg_loss=float(ls.mean()) if len(ls) else 0)

def sorted_daily(tr):
    by = {}
    for t in tr: by[t["date"]] = by.get(t["date"], 0) + t["ret"]
    return [by[k] for k in sorted(by)]

if __name__ == "__main__":
    import sys
    data = os.environ.get("NQ_DATA", "data")
    days = pickle.load(open(os.path.join(data, "days.pkl"), "rb"))
    split = "2022-01-01"
    res = {}
    keys = sys.argv[1:] or list(STRATS)
    for k in keys:
        name, fam, fn = STRATS[k]
        tr = run(days, fn)
        ins = [t for t in tr if t["date"] < split]; oos = [t for t in tr if t["date"] >= split]
        nd_in = sum(1 for d in days if str(d["date"]) < split); nd_out = len(days) - nd_in
        years = {}
        for t in tr: years.setdefault(t["date"][:4], []).append(t)
        res[k] = dict(name=name, family=fam, all=metrics(tr, len(days)), ins=metrics(ins, nd_in), oos=metrics(oos, nd_out),
                      years={y: dict(n=len(v), pts=sum(t["pts"] for t in v), ret=sum(t["ret"] for t in v),
                                     win=sum(t["pts"] > 0 for t in v) / len(v)) for y, v in sorted(years.items())},
                      trades=tr)
        a, i_, o_ = res[k]["all"], res[k]["ins"], res[k]["oos"]
        print(f"{k:11s} n={a['n']:5d} win={a['win']:.0%} PF={a['pf']:.2f} avgPts={a['avg_pts']:6.2f} "
              f"Sharpe={a['sharpe']:5.2f} | IS PF={i_['pf']:.2f} Sh={i_['sharpe']:5.2f} | OOS PF={o_['pf']:.2f} Sh={o_['sharpe']:5.2f} "
              f"| +yrs={sum(v['pts']>0 for v in res[k]['years'].values())}/{len(res[k]['years'])}", flush=True)
    pickle.dump(res, open(os.path.join(data, f"res_{os.environ.get('TAG','base')}.pkl"), "wb"))
