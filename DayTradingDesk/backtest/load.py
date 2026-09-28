"""Load NQ 1-minute bars, convert to New York time, and build per-day RTH arrays."""
import glob, os
import numpy as np, pandas as pd

DATA = os.environ.get("NQ_DATA", "data")

def load_minutes():
    fs = sorted(glob.glob(os.path.join(DATA, "NQ_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    df = df.drop(columns="timestamp").drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df

def build_days(df):
    """Return list of dicts, one per trading day with a full RTH session (9:30-16:00 ET)."""
    ts = df["ts"]
    mins = ts.dt.hour * 60 + ts.dt.minute
    df = df.assign(date=ts.dt.date, m=mins)
    # Globex session date: bars from 18:00 belong to the next trading day
    sess = ts.dt.normalize() + pd.to_timedelta((mins >= 18 * 60).astype(int), unit="D")
    df["sess"] = sess.dt.date
    days = []
    rth_idx = np.arange(570, 960)  # 9:30 .. 15:59
    prev = None
    for sd, g in df.groupby("sess", sort=True):
        r = g[(g.m >= 570) & (g.m < 960)]
        if len(r) < 370:          # skip half days / holidays / gaps
            prev = None if len(r) == 0 else prev
            continue
        r = r.set_index("m").reindex(rth_idx)
        r[["open", "high", "low", "close"]] = r[["open", "high", "low", "close"]].ffill().bfill()
        r["volume"] = r["volume"].fillna(0)
        on = g[g.m < 570]  # overnight portion (18:00 prior evening .. 9:29)
        on = on[(on.ts.dt.date < sd) | (on.m < 570)]
        d = dict(date=sd,
                 o=r.open.values, h=r.high.values, l=r.low.values, c=r.close.values, v=r.volume.values.astype(float),
                 onh=on.high.max() if len(on) else np.nan, onl=on.low.min() if len(on) else np.nan,
                 on_open=on.open.iloc[0] if len(on) else np.nan)
        if prev is not None:
            d["pdh"], d["pdl"], d["pdc"] = prev["h"].max(), prev["l"].min(), prev["c"][-1]
            # roll detection: overnight open vs previous RTH close jump > 1.5% with no news is almost always a contract roll
            d["roll"] = bool(np.isfinite(d["on_open"]) and abs(d["on_open"] / d["pdc"] - 1) > 0.015 and abs(d["o"][0] / d["on_open"] - 1) < 0.01)
        else:
            d["pdh"] = d["pdl"] = d["pdc"] = np.nan; d["roll"] = True
        days.append(d); prev = d
    return days

if __name__ == "__main__":
    df = load_minutes()
    print("rows", len(df), df.ts.min(), df.ts.max())
    days = build_days(df)
    print("RTH days", len(days), "roll-flagged", sum(d["roll"] for d in days))
    import pickle; pickle.dump(days, open(os.path.join(DATA, "days.pkl"), "wb"))
    import collections
    print(collections.Counter(d["date"].year for d in days))
    gaps = sorted(((d["o"][0] / d["pdc"] - 1) * 100, d["date"]) for d in days if np.isfinite(d["pdc"]))
    print("largest gaps", gaps[:5], gaps[-5:])
