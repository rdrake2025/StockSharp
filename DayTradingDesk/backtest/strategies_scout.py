"""Round 3: strategies the Scout agent found in published research, implemented exactly as specified in its report.

Calendar rules use a trading calendar built from the minute data (half-days included), so month-end / holiday
offsets are correct even though half-day sessions are not traded.
"""
import os, pickle, datetime as dt
import numpy as np
import pandas as pd
from bt import TICK, EOD, trade

D = os.environ.get("NQ_DATA", "data")

# Scheduled 2:00 pm ET FOMC statement days (federalreserve.gov calendars, compiled by the Scout agent)
FOMC = set("""2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10""".split())

def _build_calendar():
    cache = os.path.join(D, "scout_cal.pkl")
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    from load import load_minutes
    df = load_minutes(); ts = df["ts"]; m = ts.dt.hour * 60 + ts.dt.minute
    rth = df[(m >= 570) & (m < 960)]
    cnt = rth.groupby(rth["ts"].dt.date).size()
    tdays = sorted(str(d) for d, c in cnt.items() if c >= 150)
    # overnight prices per session date: 23:30 prior evening -> 03:30, plus 02:00 -> 03:00
    df = df.assign(m=m, date=ts.dt.date)
    ov = {}
    eve = df[(df.m >= 23 * 60 + 30)]            # 23:30-23:59 belongs to the NEXT session
    mor = df[(df.m >= 0) & (df.m <= 3 * 60 + 30)]
    first_eve = eve.groupby("date").first()
    for d, g in mor.groupby("date"):
        prev = d - dt.timedelta(days=1)
        if prev.weekday() == 5: prev = prev - dt.timedelta(days=1)   # Sunday evening session for Monday
        if prev not in first_eve.index: continue
        e = first_eve.loc[prev]
        g = g.sort_values("ts")
        x = g[g.m <= 210]
        if len(x) < 100: continue
        a = g[g.m >= 120]; b = g[g.m <= 180]
        ov[str(d)] = dict(o2330=float(e.open), c0330=float(x.close.iloc[-1]),
                          lo=float(min(e.low, x.low.min())), hi=float(max(e.high, x.high.max())),
                          o0200=float(a.open.iloc[0]) if len(a) else np.nan, c0300=float(b.close.iloc[-1]) if len(b) else np.nan,
                          lo2=float(g[(g.m >= 120) & (g.m <= 180)].low.min()) if len(a) else np.nan)
    out = dict(tdays=tdays, ov=ov)
    pickle.dump(out, open(cache, "wb"))
    return out

def _easter(y):
    a = y % 19; b, c = divmod(y, 100); d_, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d_ - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; mo, da = divmod(h + l - 7 * m + 114, 31)
    return dt.date(y, mo, da + 1)

def nyse_holidays(y):
    """NYSE full-day closures (CME often still trades a short session on these, so they can't be read off the data)."""
    def nth(month, weekday, n):
        x = dt.date(y, month, 1); x += dt.timedelta(days=(weekday - x.weekday()) % 7); return x + dt.timedelta(days=7 * (n - 1))
    def last(month, weekday):
        x = dt.date(y, month + 1, 1) - dt.timedelta(days=1); return x - dt.timedelta(days=(x.weekday() - weekday) % 7)
    def obs(x):
        return x - dt.timedelta(days=1) if x.weekday() == 5 else x + dt.timedelta(days=1) if x.weekday() == 6 else x
    h = {nth(1, 0, 3), nth(2, 0, 3), _easter(y) - dt.timedelta(days=2), last(5, 0), obs(dt.date(y, 7, 4)),
         nth(9, 0, 1), nth(11, 3, 4), obs(dt.date(y, 12, 25))}
    ny = dt.date(y, 1, 1)
    if ny.weekday() != 5: h.add(obs(ny))            # NYSE does not observe New Year's on Friday Dec 31
    if y >= 2022: h.add(obs(dt.date(y, 6, 19)))
    h |= {x for x in (dt.date(2018, 12, 5), dt.date(2025, 1, 9)) if x.year == y}   # national days of mourning
    return {str(x) for x in h}

CAL = _build_calendar()
_HOL = set().union(*(nyse_holidays(y) for y in range(2014, 2027)))
TD = [d for d in CAL["tdays"] if d not in _HOL and dt.date.fromisoformat(d).weekday() < 5]
TI = {d: i for i, d in enumerate(TD)}
OV = CAL["ov"]

def _month_info():
    """For every trading day: offset to its month's last trading day (0 = T, -1 = T-1 ...),
    and 1-based index after the previous month's T (1 = first trading day of the month)."""
    info = {}
    by_m = {}
    for d in TD: by_m.setdefault(d[:7], []).append(d)
    for mth, ds in by_m.items():
        for k, d in enumerate(ds):
            info[d] = dict(to_T=k - (len(ds) - 1), nth=k + 1, month_days=ds)
    return info
MI = _month_info()

def settle_k(date):
    """Trading days between the settlement-deadline day S and month-end T (Etula et al.): S = T-k."""
    if date < "2017-09-05": return 3
    if date < "2024-05-28": return 2
    return 1

def rth_trade(d, side, i=0, end=EOD):
    e = d["o"][i]
    return trade(d, i, side, e, e - side * 1e6, end=end)

# ---------------------------------------------------------------- strategies
def fomc_pre():
    """#1 Lucca & Moench (2015): long on FOMC statement days 09:30 -> 13:55 ET."""
    def f(d, ctx):
        return rth_trade(d, 1, 0, 264) if str(d["date"]) in FOMC else None
    return f

def fomc_post():
    """#2a Citi statistic: short on FOMC days 14:05 -> close."""
    def f(d, ctx):
        return rth_trade(d, -1, 275, EOD) if str(d["date"]) in FOMC else None
    return f

def fomc_fade():
    """#2b Narain & Sangani (2026): at 14:30 fade the 14:00-14:30 move until 15:30; stop at the 14:00-14:30 extreme."""
    def f(d, ctx):
        if str(d["date"]) not in FOMC: return None
        mv = d["c"][299] - d["o"][270]
        if abs(mv) < 1: return None
        side = -1 if mv > 0 else 1
        st = d["h"][270:300].max() + TICK if side < 0 else d["l"][270:300].min() - TICK
        return trade(d, 300, side, d["o"][300], st, end=359)
    return f

def _ov_trade(d, side, a, b, lo, hi):
    """Overnight trade recorded against the session date (flat before RTH)."""
    from bt import COST
    pts = (b - a) * side - COST
    adverse = (lo - a) if side > 0 else (a - hi)
    return dict(date=str(d["date"]), side=side, i=0, j=0, entry=float(a), exit=float(b), stop0=None, risk=None,
                pts=float(pts), R=None, ret=float(pts / a), why="overnight", mae=float(min(0.0, adverse) - COST))

def overnight_drift(conditional=False):
    """#3 Bondarenko & Muravyev (2023): long 23:30 -> 03:30 ET. Conditional (Boyarchenko et al.): only after
    a prior RTH session in the bottom tercile of the trailing 60 sessions."""
    def f(d, ctx):
        x = OV.get(str(d["date"]))
        if not x: return None
        if conditional:
            h = ctx["hist"]
            if len(h) < 40: return None
            r = [y["c"][-1] / y["o"][0] - 1 for y in h[-40:]]
            if r[-1] > np.percentile(r, 33.3): return None
        return _ov_trade(d, 1, x["o2330"], x["c0330"], x["lo"], x["hi"])
    return f

def dash_for_cash(long_only=False):
    """#4 Etula, Rinne, Suominen & Vaittinen (2020): short RTH on S-5..S-1, long on S..T+3, with S = T-k by settlement cycle."""
    def f(d, ctx):
        ds = str(d["date"]); mi = MI.get(ds)
        if not mi: return None
        k = settle_k(ds); side = 0
        if -k <= mi["to_T"] <= 0: side = 1
        elif mi["nth"] <= 3: side = 1
        elif -(k + 5) <= mi["to_T"] <= -(k + 1): side = -1
        if side == 0 or (long_only and side < 0): return None
        return rth_trade(d, side)
    return f

def rebal_6040():
    """#5 Harvey, Mazzoleni & Melone (2025) proxy: month-to-date NQ return > 0 -> short T-4..T-1, long on T (reverse if < 0)."""
    prior_T_close = {}
    def f(d, ctx):
        ds = str(d["date"]); mi = MI.get(ds); h = ctx["hist"]
        if not mi or not h: return None
        if not (-4 <= mi["to_T"] <= 0): return None
        mth = ds[:7]
        base = next((y for y in reversed(h) if str(y["date"])[:7] < mth), None)
        if base is None: return None
        sig = h[-1]["c"][-1] / base["c"][-1] - 1
        if sig == 0: return None
        side = (-1 if sig > 0 else 1) if mi["to_T"] < 0 else (1 if sig > 0 else -1)
        return rth_trade(d, side)
    return f

def turn_of_month():
    """#6 Lakonishok & Smidt / McConnell & Xu: long RTH on the last trading day and first three of the new month."""
    def f(d, ctx):
        mi = MI.get(str(d["date"]))
        return rth_trade(d, 1) if mi and (mi["to_T"] == 0 or mi["nth"] <= 3) else None
    return f

FOMC_I = sorted(TI[x] for x in FOMC if x in TI)
def fomc_week(ds):
    i = TI.get(ds)
    if i is None: return None
    nxt = next((j for j in FOMC_I if j >= i), None)
    if nxt is not None and nxt - i == 1: return 0          # day -1 belongs to week 0
    prv = next((j for j in reversed(FOMC_I) if j <= i), None)
    if prv is None: return None
    o = i - prv
    return 0 if o <= 3 else 1 + (o - 4) // 5

def fomc_cycle():
    """#7 Cieslak, Morse & Vissing-Jorgensen (2019): long RTH in even FOMC-cycle weeks (0, 2, 4, 6)."""
    def f(d, ctx):
        w = fomc_week(str(d["date"]))
        return rth_trade(d, 1) if w is not None and w % 2 == 0 else None
    return f

def open_reversal():
    """#9 Grant, Wolf & Yu (2005) with the Scout's default: 9:30-9:45 move beyond 2 sd of the last 60 sessions -> fade to the close."""
    def f(d, ctx):
        h = ctx["hist"]
        if len(h) < 40: return None
        ms = np.array([y["c"][14] / y["o"][0] - 1 for y in h[-40:]])
        m = d["c"][14] / d["o"][0] - 1
        if abs(m) <= 2 * ms.std(): return None
        side = -1 if m > 0 else 1
        st = d["h"][:15].max() + TICK if side < 0 else d["l"][:15].min() - TICK
        return trade(d, 15, side, d["o"][15], st)
    return f

def _third_friday(y, m):
    x = dt.date(y, m, 1); x += dt.timedelta(days=(4 - x.weekday()) % 7); return x + dt.timedelta(days=14)

def opex_week():
    """#10 Stivers & Sun (2013): long RTH Monday-Thursday of the week containing the monthly option expiration."""
    def f(d, ctx):
        x = d["date"]; tf = _third_friday(x.year, x.month)
        return rth_trade(d, 1) if (tf - x).days in (1, 2, 3, 4) and x.weekday() <= 3 else None
    return f

def pre_holiday():
    """#11 Ariel (1990): long RTH on the last trading day before an exchange holiday (weekday closure)."""
    def f(d, ctx):
        ds = str(d["date"]); i = TI.get(ds)
        if i is None or i + 1 >= len(TD): return None
        x = d["date"]; nb = x + dt.timedelta(days=1)
        while nb.weekday() >= 5: nb += dt.timedelta(days=1)
        return rth_trade(d, 1) if TD[i + 1] > str(nb) else None
    return f

def monday_open():
    """#12 Harris (1986): short the first 45 minutes of the first trading day of the week."""
    def f(d, ctx):
        h = ctx["hist"]
        if not h or (d["date"] - h[-1]["date"]).days < 3: return None
        return rth_trade(d, -1, 0, 44)
    return f

SCOUT = {
    "fomc_pre":   ("Pre-FOMC drift (long 9:30–1:55 on Fed days)", "Calendar", fomc_pre()),
    "fomc_post":  ("FOMC post-statement short (2:05 pm–close)", "Calendar", fomc_post()),
    "fomc_fade":  ("FOMC press-conference fade (2:30–3:30)", "Reversion", fomc_fade()),
    "overnight":  ("Overnight EU-open drift (11:30 pm–3:30 am)", "Calendar", overnight_drift(False)),
    "overnight_c":("Overnight drift after a weak day", "Calendar", overnight_drift(True)),
    "dash4cash":  ("Month-end ‘dash for cash’ (long/short)", "Calendar", dash_for_cash(False)),
    "dash4cash_l":("Month-end ‘dash for cash’ (long leg only)", "Calendar", dash_for_cash(True)),
    "rebal":      ("60/40 rebalancing front-run (proxy)", "Calendar", rebal_6040()),
    "tom":        ("Turn of the month (long)", "Calendar", turn_of_month()),
    "fomc_cycle": ("FOMC-cycle even weeks (long)", "Calendar", fomc_cycle()),
    "open_rev":   ("Large opening-move reversal", "Reversion", open_reversal()),
    "opex":       ("Option-expiration week (long Mon–Thu)", "Calendar", opex_week()),
    "preholiday": ("Pre-holiday (long)", "Calendar", pre_holiday()),
    "monday":     ("Monday-open short (9:30–10:15)", "Calendar", monday_open()),
}

LEDGER = [
    dict(key="fomc_pre", name="Pre-FOMC announcement drift", source="Lucca & Moench (2015), Journal of Finance; Kurov, Wolfe & Gilbert (2021)", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7525326/", claim="+49 bp in the 24 h before FOMC statements (1994–2011), fading to about +9 bp in 2016–19."),
    dict(key="fomc_post", name="FOMC post-statement drift", source="Citi strategists via Investing.com", url="https://in.investing.com/news/stock-market-news/trade-the-fomc-how-stocks-tend-to-perform-on-fed-hike-days-5594832", claim="Gains right after the statement usually reverse during the press conference; declines extend."),
    dict(key="fomc_fade", name="Press-conference fade", source="Narain & Sangani (2026), International Journal of Central Banking", url="https://www.ijcb.org/journal/v22n1/market-impact-fed-communications-role-press-conference", claim="Powell-era press conferences are 3× more volatile and often reverse the statement reaction."),
    dict(key="overnight", name="Overnight EU-open drift", source="Bondarenko & Muravyev (2023), JFQA; Boyarchenko, Larsen & Whelan (2023), RFS", url="https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/", claim="ES 23:30–03:30 ET earned 7.6%/yr (2004–18); the NY Fed reports it vanished after 2021."),
    dict(key="overnight_c", name="Overnight drift after selloffs", source="Boyarchenko, Larsen & Whelan (2023), Review of Financial Studies", url="https://www.newyorkfed.org/research/staff_reports/sr917", claim="The drift is strongest after RTH selloffs (negative closing imbalance)."),
    dict(key="dash4cash", name="Month-end dash for cash", source="Etula, Rinne, Suominen & Vaittinen (2020), Review of Financial Studies", url="https://academic.oup.com/rfs/article/33/1/75/5494694", claim="Selling pressure before the month-end settlement deadline, then reversal. Negative correlation in 25 of 25 markets."),
    dict(key="dash4cash_l", name="Dash for cash, long leg", source="Etula et al. (2020)", url="https://academic.oup.com/rfs/article/33/1/75/5494694", claim="The reversal days returned +103% cumulative excess in 2003–2013."),
    dict(key="rebal", name="60/40 rebalancing front-run", source="Harvey, Mazzoleni & Melone (2025), NBER w33554", url="https://www.nber.org/papers/w33554", claim="When stocks are overweight vs 60/40, next-day returns are −17 bp. ES/TY Sharpe above 1 (uses bonds; NQ-only proxy here)."),
    dict(key="tom", name="Turn of the month", source="McConnell & Xu (2008), Financial Analysts Journal", url="https://business.purdue.edu/faculty/mcconnell/publications/Equity-Returns-at-the-Turn-of-the-Month.pdf", claim="0.15%/day on days −1…+3 (1926–2005); recent tests find it has faded."),
    dict(key="fomc_cycle", name="FOMC-cycle even weeks", source="Cieslak, Morse & Vissing-Jorgensen (2019), Journal of Finance", url="https://faculty.haas.berkeley.edu/morse/research/papers/cycle_paper_cieslak_morse_vissingjorgensen.pdf", claim="Since 1994 the equity premium was earned in even weeks of the FOMC cycle."),
    dict(key="open_rev", name="Large opening-move reversal", source="Grant, Wolf & Yu (2005), Journal of Banking & Finance", url="https://www.sciencedirect.com/science/article/abs/pii/S0378426604000949", claim="S&P futures reversed large opening moves (1987–2002), but the edge was near trading costs."),
    dict(key="opex", name="Option-expiration week", source="Stivers & Sun (2013), Journal of Banking & Finance", url="https://quantpedia.com/strategies/option-expiration-week-effect", claim="High returns in expiration week as dealers unwind hedges (1988–2010)."),
    dict(key="preholiday", name="Pre-holiday effect", source="Ariel (1990), Journal of Finance", url="https://quantpedia.com/strategies/pre-holiday-effect", claim="Returns before holidays were 9–14× normal historically; recent studies find it insignificant."),
    dict(key="monday", name="Monday-open weakness", source="Harris (1986), Journal of Financial Economics", url=None, claim="Monday's negative return accrued in the first 45 minutes (NYSE 1981–83)."),
    dict(key=None, name="Macro announcement-day premium (CPI, jobs)", source="Savor & Wilson (2013), JFQA", url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1312091", claim="11.4 bp excess on announcement days vs 1.1 bp otherwise.", status="Needs the BLS release calendar (not in this data)"),
]
