"""Luck baseline: noise-band trades with random directions, run through the same LucidFlex rules."""
import os, pickle, json, copy
import numpy as np
import bt, lucid
D = os.environ["NQ_DATA"]
days = pickle.load(open(os.path.join(D, "days.pkl"), "rb")); dates = [str(d["date"]) for d in days]
res = pickle.load(open(os.path.join(D, "res_all.pkl"), "rb"))
out = {}
for seed in range(5):
    rng = np.random.default_rng(seed); tr = []
    for t in res["noise"]["trades"]:
        s = rng.choice([-1, 1]); g = t["pts"] + bt.COST; t2 = dict(t)
        if s < 0:  # flipped trade: gross reverses; adverse excursion ~ favourable excursion of the original (approx: |gross| or MAE mirror)
            t2["pts"] = -g - bt.COST; t2["mae"] = min(0.0, -max(g, 0) - bt.COST) if g > 0 else t["mae"]
        tr.append(t2)
    for r in lucid.study(lucid.day_series(tr, dates), sizes=("50K", "150K"), micro_opts=(1, 2, 4, 6, 10, 20)):
        key = f"{r['size']}|{r['micros']}"; out.setdefault(key, []).append(r)
agg = {k: dict(size=v[0]["size"], micros=v[0]["micros"], pass_rate=round(float(np.mean([x["pass_rate"] for x in v])), 3),
               fail_rate=round(float(np.mean([x["fail_rate"] for x in v])), 3), funded_paid=round(float(np.mean([x["funded_paid"] for x in v]))),
               ev=round(float(np.mean([x["ev"] for x in v])))) for k, v in out.items()}
json.dump(agg, open(os.path.join(D, "lucid_luck.json"), "w"))
for k, v in agg.items(): print(k, v)
