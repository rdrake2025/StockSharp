"""Additional intraday strategies (round 2). Same execution conventions as bt.py."""
import numpy as np
from bt import TICK, EOD, five, ema, trade, first_cross_up, _mk, orb, orb5_zarattini

def on_breakout():
    """Overnight range breakout: first 5-min close beyond the Globex high/low (9:35-12:00), hold to close.
    Stop at the far side of the breakout bar (min 10 pts)."""
    def f(d, ctx):
        if d["roll"] or not np.isfinite(d["onh"]): return None
        b = five(d); H, L, C = b["H"], b["L"], b["C"]
        onh, onl = d["onh"], d["onl"]
        if not (onl < d["o"][0] < onh): return None
        for k in range(1, 30):
            if C[k] > onh:
                e = d["o"][5 * (k + 1)]; st = min(L[k], e - 10)
                return trade(d, 5 * (k + 1), 1, e, st)
            if C[k] < onl:
                e = d["o"][5 * (k + 1)]; st = max(H[k], e + 10)
                return trade(d, 5 * (k + 1), -1, e, st)
        return None
    return f

def opening_drive(th=0.25):
    """Opening drive: if 9:30-10:00 moved more than th% from the open, join at 10:00, stop at the 9:30 open, hold to close."""
    def f(d, ctx):
        r = (d["c"][29] / d["o"][0] - 1) * 100
        if abs(r) < th: return None
        side = 1 if r > 0 else -1
        e = d["o"][30]; st = d["o"][0]
        return trade(d, 30, side, e, st)
    return f

def vwap_trend():
    """VWAP trend-follow: at 10:30 take the side of VWAP price is on (if also beyond the open);
    exit on the first 5-min close back across VWAP, or at the close."""
    def f(d, ctx):
        b = five(d); C, VW = b["C"], b["VW"]
        k0 = 11  # 5-min bar ending 10:30
        side = 1 if (C[k0] > VW[k0] and C[k0] > d["o"][0]) else -1 if (C[k0] < VW[k0] and C[k0] < d["o"][0]) else 0
        if not side: return None
        i = 5 * (k0 + 1); e = d["o"][i]
        for k in range(k0 + 1, 78):
            if (C[k] - VW[k]) * side < 0:
                j = min(EOD, 5 * (k + 1))
                ex = d["o"][j] if j < EOD else d["c"][EOD]
                return _mk(d, side, i, j, e, ex, "vwap")
        return _mk(d, side, i, EOD, e, d["c"][EOD], "eod")
    return f

def nr_orb(n=4):
    """Crabel NR-n filter: 30-min ORB (2R) only on days after the narrowest daily range of the last n days."""
    base = orb(30, 2.0)
    def f(d, ctx):
        h = ctx["hist"]
        if len(h) < n: return None
        rng = [x["h"].max() - x["l"].min() for x in h[-n:]]
        if rng[-1] > min(rng): return None
        return base(d, ctx)
    return f

def bollinger_break(n=20, k_sd=2.0):
    """5-min Bollinger breakout after 10:00: close outside the band -> join, exit on a close through the middle band."""
    def f(d, ctx):
        C = five(d)["C"]
        for k in range(n, 72):
            w = C[k - n + 1:k + 1]; m, sd = w.mean(), w.std()
            if sd <= 0: continue
            side = 1 if C[k] > m + k_sd * sd else -1 if C[k] < m - k_sd * sd else 0
            if not side or k < 6: continue
            i = 5 * (k + 1); e = d["o"][i]
            for q in range(k + 1, 78):
                w2 = C[q - n + 1:q + 1]; m2 = w2.mean()
                if (C[q] - m2) * side < 0:
                    j = min(EOD, 5 * (q + 1)); ex = d["o"][j] if j < EOD else d["c"][EOD]
                    return _mk(d, side, i, j, e, ex, "mid")
            return _mk(d, side, i, EOD, e, d["c"][EOD], "eod")
        return None
    return f

def rsi2_reversion():
    """Connors-style RSI(2) on 5-min bars: RSI < 10 -> buy, > 90 -> sell (10:00-15:00); exit when RSI crosses 50,
    at the close, or at a stop of 1.5x the average 5-min range."""
    def f(d, ctx):
        b = five(d); C, H, L = b["C"], b["H"], b["L"]
        dlt = np.diff(C, prepend=C[0]); up = np.where(dlt > 0, dlt, 0); dn = np.where(dlt < 0, -dlt, 0)
        au, ad = ema(up, 3), ema(dn, 3)  # 2-period Wilder ~ 3-period EMA
        rsi = 100 - 100 / (1 + au / np.maximum(ad, 1e-9))
        atr = np.mean(H[:6] - L[:6])
        for k in range(6, 66):
            side = 1 if rsi[k] < 10 else -1 if rsi[k] > 90 else 0
            if not side: continue
            i = 5 * (k + 1); e = d["o"][i]; st = e - side * 1.5 * atr
            ex_i = None
            for q in range(k + 1, 78):
                if (side > 0 and rsi[q] > 50) or (side < 0 and rsi[q] < 50): ex_i = 5 * (q + 1); break
            return trade(d, i, side, e, st, end=min(EOD, ex_i) if ex_i else EOD)
        return None
    return f

def lunch_break():
    """Lunch range breakout: range of 11:30-13:30; stop-entry on the first break after 13:30, stop other side, hold to close."""
    def f(d, ctx):
        a, z = 120, 240
        hi, lo = d["h"][a:z].max(), d["l"][a:z].min()
        if hi - lo < 8: return None
        iu = first_cross_up(d["h"], hi, z); idn = first_cross_up(-d["l"], -lo, z)
        if iu is None and idn is None: return None
        if iu is not None and (idn is None or iu < idn):
            if iu > 360: return None
            return trade(d, iu, 1, hi + TICK, lo, stop_entry=True)
        if idn > 360: return None
        return trade(d, idn, -1, lo - TICK, hi, stop_entry=True)
    return f

def late_momentum():
    """Late-day momentum: at 15:00 trade in the direction of the day so far (vs the 9:30 open) until 16:00."""
    def f(d, ctx):
        r = d["c"][329] - d["o"][0]
        if abs(r) < 1: return None
        side = 1 if r > 0 else -1
        return trade(d, 330, side, d["o"][330], d["o"][330] - side * 1e6)
    return f

def orb_fade():
    """Failed ORB: price breaks the 30-min range, then a 5-min bar closes back inside -> fade to the other side."""
    def f(d, ctx):
        b = five(d); H, L, C = b["H"], b["L"], b["C"]
        hi, lo = H[:6].max(), L[:6].min()
        bu = bd = False; hx, lx = -1e9, 1e9
        for k in range(6, 48):
            if H[k] > hi + TICK: bu = True
            if L[k] < lo - TICK: bd = True
            hx = max(hx, H[k]); lx = min(lx, L[k])
            if bu and C[k] < hi:
                i = 5 * (k + 1); e = d["o"][i]; st = hx + TICK
                return trade(d, i, -1, e, st, lo)
            if bd and C[k] > lo:
                i = 5 * (k + 1); e = d["o"][i]; st = lx - TICK
                return trade(d, i, 1, e, st, hi)
        return None
    return f

def orb5_gap_aligned():
    """5-min first-candle ORB, only when the candle agrees with the overnight gap direction."""
    base = orb5_zarattini(10)
    def f(d, ctx):
        if d["roll"] or not np.isfinite(d["pdc"]): return None
        g = d["o"][0] - d["pdc"]; cdir = d["c"][4] - d["o"][0]
        if g * cdir <= 0: return None
        return base(d, ctx)
    return f

def channel_break(n=12):
    """1-hour Donchian breakout on 5-min bars after 10:30: close beyond the prior 12-bar high/low -> join;
    exit on a close beyond the opposite 6-bar channel, or at the close. One trade per day."""
    def f(d, ctx):
        b = five(d); H, L, C = b["H"], b["L"], b["C"]
        for k in range(max(n, 12), 66):
            side = 1 if C[k] > H[k - n:k].max() else -1 if C[k] < L[k - n:k].min() else 0
            if not side: continue
            i = 5 * (k + 1); e = d["o"][i]
            for q in range(k + 1, 78):
                if (side > 0 and C[q] < L[q - 6:q].min()) or (side < 0 and C[q] > H[q - 6:q].max()):
                    j = min(EOD, 5 * (q + 1)); ex = d["o"][j] if j < EOD else d["c"][EOD]
                    return _mk(d, side, i, j, e, ex, "channel")
            return _mk(d, side, i, EOD, e, d["c"][EOD], "eod")
        return None
    return f

def on_fade():
    """Overnight-extreme fade: first touch of the Globex high/low after 9:35 that fails (5-min close back inside) -> fade to VWAP."""
    def f(d, ctx):
        if d["roll"] or not np.isfinite(d["onh"]): return None
        b = five(d); H, L, C, VW = b["H"], b["L"], b["C"], b["VW"]
        onh, onl = d["onh"], d["onl"]
        if not (onl < d["o"][0] < onh): return None
        tu = td = False; hx, lx = -1e9, 1e9
        for k in range(1, 30):
            if H[k] > onh: tu = True
            if L[k] < onl: td = True
            hx = max(hx, H[k]); lx = min(lx, L[k])
            if tu and C[k] < onh and VW[k] < C[k] - 4:
                i = 5 * (k + 1); return trade(d, i, -1, d["o"][i], hx + TICK, VW[k])
            if td and C[k] > onl and VW[k] > C[k] + 4:
                i = 5 * (k + 1); return trade(d, i, 1, d["o"][i], lx - TICK, VW[k])
        return None
    return f

MORE = {
    "on_break":  ("Overnight high/low breakout", "Breakout", on_breakout()),
    "on_fade":   ("Overnight high/low failed test", "Reversion", on_fade()),
    "drive":     ("Opening drive continuation", "Momentum", opening_drive(0.25)),
    "vwap_trend":("VWAP trend-follow from 10:30", "Momentum", vwap_trend()),
    "nr4_orb":   ("30-min ORB after NR4 day", "Breakout", nr_orb(4)),
    "bb_break":  ("Bollinger breakout (5-min)", "Breakout", bollinger_break()),
    "rsi2":      ("RSI(2) reversion (5-min)", "Reversion", rsi2_reversion()),
    "lunch":     ("Lunch-range breakout", "Breakout", lunch_break()),
    "late_mom":  ("Late-day momentum (3–4 pm)", "Momentum", late_momentum()),
    "orb_fade":  ("Failed 30-min ORB fade", "Reversion", orb_fade()),
    "orb5_gap":  ("5-min ORB aligned with gap", "Breakout", orb5_gap_aligned()),
    "channel":   ("1-hour channel breakout", "Breakout", channel_break()),
}
