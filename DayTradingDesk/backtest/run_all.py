"""Run every strategy (bt.STRATS + strategies_more.MORE) and save results for proof.py / lucid.py / export.py."""
import os, pickle
import bt
from strategies_more import MORE

def main():
    data = os.environ.get("NQ_DATA", "data")
    days = pickle.load(open(os.path.join(data, "days.pkl"), "rb"))
    split = "2022-01-01"
    nd_in = sum(1 for d in days if str(d["date"]) < split); nd_out = len(days) - nd_in
    allS = {**bt.STRATS, **MORE}; res = {}
    for k, (name, fam, fn) in allS.items():
        tr = bt.run(days, fn)
        ins = [t for t in tr if t["date"] < split]; oos = [t for t in tr if t["date"] >= split]
        years = {}
        for t in tr: years.setdefault(t["date"][:4], []).append(t)
        res[k] = dict(name=name, family=fam, new=k in MORE, all=bt.metrics(tr, len(days)), ins=bt.metrics(ins, nd_in), oos=bt.metrics(oos, nd_out),
                      years={y: dict(n=len(v), pts=sum(t["pts"] for t in v), ret=sum(t["ret"] for t in v), win=sum(t["pts"] > 0 for t in v) / len(v)) for y, v in sorted(years.items())},
                      trades=tr)
        a, o = res[k]["all"], res[k]["oos"]
        print(f"{'*' if k in MORE else ' '}{k:11s} n={a['n']:5d} win={a['win']:.0%} PF={a['pf']:.2f} avg={a['avg_pts']:6.2f} Sh={a['sharpe']:5.2f} OOS PF={o.get('pf',0):.2f} Sh={o.get('sharpe',0):5.2f} +yrs={sum(v['pts']>0 for v in res[k]['years'].values())}/{len(res[k]['years'])}", flush=True)
    pickle.dump(res, open(os.path.join(data, "res_all.pkl"), "wb"))

if __name__ == "__main__":
    main()
