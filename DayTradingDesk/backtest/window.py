"""Noise-band momentum restricted to a personal trading window: entries only at checks inside the window,
everything flat at the window's end. Tests whether the edge survives a 10:00-12:00 ET schedule."""
import os, pickle, math, json
import numpy as np
import bt

def noise_window(first=29, last_entry=119, flat=150, lookback=14, vm=1.0):
    """first/last_entry = 1-min bar index of the first/last check that may open a trade (29 = 10:00 close, 119 = 11:30);
    flat = bar index whose OPEN we exit at (150 = 12:00). Trailing exit rule unchanged at checks before `flat`."""
    def f(d, ctx):
        hist = ctx["hist"]
        if len(hist) < lookback or not np.isfinite(d["pdc"]) or d["roll"]: return None
        sig = np.mean([np.abs(x["c"] / x["o"][0] - 1) for x in hist[-lookback:]], axis=0) * vm
        op = d["o"][0]; ub = max(op, d["pdc"]) * (1 + sig); lb = min(op, d["pdc"]) * (1 - sig)
        vw = bt.five(d)["vw1"]; c = d["c"]; out = []; pos = 0; entry = i_in = None
        for t in range(first, flat, 30):
            if pos:
                stop = max(ub[t], vw[t]) if pos > 0 else min(lb[t], vw[t])
                if (pos > 0 and c[t] < stop) or (pos < 0 and c[t] > stop):
                    out.append(bt._mk(d, pos, i_in, t + 1, entry, d["o"][t + 1], "trail")); pos = 0
            if not pos and t <= last_entry:
                if c[t] > ub[t]: pos, entry, i_in = 1, d["o"][t + 1], t + 1
                elif c[t] < lb[t]: pos, entry, i_in = -1, d["o"][t + 1], t + 1
        if pos: out.append(bt._mk(d, pos, i_in, flat, entry, d["o"][flat], "window end"))
        return out or None
    return f

if __name__ == "__main__":
    D = os.environ["NQ_DATA"]; days = pickle.load(open(os.path.join(D, "days.pkl"), "rb"))
    rng = np.random.default_rng(1)
    rows = {}
    for name, fn in [("full day (as tested)", bt.noise_boundary(14, 1.0)),
                     ("10:00-12:00 window", noise_window(29, 119, 150)),
                     ("10:00-12:30 window", noise_window(29, 149, 180)),
                     ("10:00-13:00 window", noise_window(29, 179, 210)),
                     ("9:30-12:00, entries from 10:00", noise_window(29, 119, 150))]:
        tr = bt.run(days, fn); m = bt.metrics(tr, len(days)); o = bt.metrics([t for t in tr if t["date"] >= "2022"], sum(str(d["date"]) >= "2022" for d in days))
        yrs = {}
        for t in tr: yrs[t["date"][:4]] = yrs.get(t["date"][:4], 0) + t["pts"]
        g = np.array([t["pts"] + bt.COST for t in tr]); act = (g - bt.COST).sum()
        sims = np.array([(g * rng.choice([-1, 1], len(g)) - bt.COST).sum() for _ in range(2000)])
        daily = {}
        for t in tr: daily[t["date"]] = daily.get(t["date"], 0) + t["ret"]
        dr = np.array(list(daily.values()) + [0] * (len(days) - len(daily)))
        tstat = dr.mean() / (dr.std(ddof=1) / math.sqrt(len(dr)))
        rows[name] = dict(n=m["n"], win=round(m["win"], 3), pf=round(m["pf"], 3), sharpe=round(m["sharpe"], 2), oos_sh=round(o["sharpe"], 2), oos_pf=round(o["pf"], 3),
                          up_years=sum(v > 0 for v in yrs.values()), years=len(yrs), tot=round(m["tot_pts"]), dd=round(m["maxdd"], 4), p_dir=float((sims >= act).mean()), t=round(tstat, 2),
                          avg_win=round(m["avg_win"], 1), avg_loss=round(m["avg_loss"], 1), per_year_mnq=round(m["tot_pts"] * 2 / 10.55))
        print(name, rows[name])
    json.dump(rows, open(os.path.join(D, "window.json"), "w"))
    pickle.dump(bt.run(days, noise_window(29, 119, 150)), open(os.path.join(D, "window_trades.pkl"), "wb"))

def noise_window_hold(last_entry=119, handoff=150, lookback=14, vm=1.0):
    """Same entries as the window version, managed normally until the handoff check (12:00). A trade still open
    then gets a fixed stop at that moment's trailing level (band/VWAP) and rides unattended to the 15:59 close."""
    def f(d, ctx):
        hist = ctx["hist"]
        if len(hist) < lookback or not np.isfinite(d["pdc"]) or d["roll"]: return None
        sig = np.mean([np.abs(x["c"] / x["o"][0] - 1) for x in hist[-lookback:]], axis=0) * vm
        op = d["o"][0]; ub = max(op, d["pdc"]) * (1 + sig); lb = min(op, d["pdc"]) * (1 - sig)
        vw = bt.five(d)["vw1"]; c, o, h, l = d["c"], d["o"], d["h"], d["l"]; out = []; pos = 0; entry = i_in = None
        for t in range(29, handoff, 30):
            if pos:
                stop = max(ub[t], vw[t]) if pos > 0 else min(lb[t], vw[t])
                if (pos > 0 and c[t] < stop) or (pos < 0 and c[t] > stop):
                    out.append(bt._mk(d, pos, i_in, t + 1, entry, o[t + 1], "trail")); pos = 0
            if not pos and t <= last_entry:
                if c[t] > ub[t]: pos, entry, i_in = 1, o[t + 1], t + 1
                elif c[t] < lb[t]: pos, entry, i_in = -1, o[t + 1], t + 1
        if pos:
            t = handoff - 1; st = max(ub[t], vw[t]) if pos > 0 else min(lb[t], vw[t])
            if (pos > 0 and c[t] <= st) or (pos < 0 and c[t] >= st):
                out.append(bt._mk(d, pos, i_in, handoff, entry, o[handoff], "handoff")); return out
            ex, why = c[bt.EOD], "close"
            for j in range(handoff, bt.EOD + 1):
                if pos > 0 and l[j] <= st: ex, why = min(st, o[j]), "stop"; break
                if pos < 0 and h[j] >= st: ex, why = max(st, o[j]), "stop"; break
            out.append(bt._mk(d, pos, i_in, j, entry, ex, why))
        return out or None
    return f
