"""Evaluation of every run under DIR (and DIR/_grid): nothing is dropped. Writes to RES:
  all_runs.csv     every run x split (validation, test, blind) x stratum: AUPRC, AUROC, top-0.5/1/5/10/20% (per day) precision and recall, Brier, ECE, and Brier / ECE after isotonic calibration fitted on 2024
  per_state.csv    every run x split x state: AUPRC, AUROC
  seeds.csv        mean, sd, min, max over seeds for every group of runs
  paired.csv       paired calendar-week block bootstrap (2,000 draws) of the AUPRC difference on the test year, on seed-averaged predictions, with the decision-rule verdict
Strata: all, season (fire-season rows), new_ignition (no detection on day t), strict (no forest detection in the last 7 days).
Usage: python src/evaluate.py --out DIR --res RES [--splits validation,test,blind]"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as auprc, roc_auc_score as auroc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import brier, ece

STRATA = {"all": lambda d: np.ones(len(d), bool), "season": lambda d: d.season.values == 1, "new_ignition": lambda d: d.had_det_t.values == 0, "strict": lambda d: d.fd7f.values == 0}
BINS = 10000
TOPQ = (0.005, 0.01, 0.05, 0.1, 0.2)


def topk_day(y, p, day, qs=TOPQ):
    """Precision and recall among the top q of cells ranked within each day (q of that day's cells, at least one), pooled over days."""
    o = np.lexsort((-p, day)); y, day = y[o], day[o]; start = np.r_[0, np.flatnonzero(np.diff(day)) + 1]; n = np.diff(np.r_[start, len(day)]); rank = np.arange(len(day)) - np.repeat(start, n); out = {}
    for q in qs:
        sel = rank < np.maximum(1, np.ceil(q * np.repeat(n, n))); k = f"{q * 100:g}".replace(".", "p"); out[f"top{k}_prec"] = y[sel].mean(); out[f"top{k}_rec"] = y[sel].sum() / max(1, y.sum())
    return out


def one(y, p, q=None, day=None):
    r = dict(n=len(y), pos=int(y.sum()))
    if y.sum() == 0 or y.sum() == len(y): return r
    r.update(auprc=auprc(y, p), auroc=auroc(y, p)); r.update(topk_day(y, p, day)); r.update(brier=brier(y, p), ece=ece(y, p))
    if q is not None: r.update(brier_iso=brier(y, q), ece_iso=ece(y, q))
    return r


def load():
    runs = []
    for mp in sorted(glob.glob(f"{A.out}/*/manifest.json") + glob.glob(f"{A.out}/_grid/*/manifest.json")):
        m = json.load(open(mp)); d = os.path.dirname(mp)
        if m.get("complete") and os.path.isdir(f"{d}/predictions"): m["dir"] = d; m["grid"] = "/_grid/" in mp or "__" in m["run_id"]; m["group"] = re.sub(r"_s\d+$", "", m["run_id"]); runs.append(m)
    return runs


def binned(y, p, week, edges, W):
    """positives and negatives per [week, score bin]; high score = high bin"""
    b = np.searchsorted(edges, p, side="right"); P = np.zeros((W, len(edges) + 1)); N = np.zeros_like(P); np.add.at(P, (week[y == 1], b[y == 1]), 1); np.add.at(N, (week[y == 0], b[y == 0]), 1); return P, N


def ap_from(P, N):
    tp = np.cumsum(P[::-1]); fp = np.cumsum(N[::-1]); prec = tp / np.maximum(tp + fp, 1); rec = tp / max(tp[-1], 1); return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def paired(a, b, draws=2000, seed=0):
    """a, b: frames with y, p, week in the same row order. Returns (difference, low, high, binning check)."""
    y = a.y.values; wk = pd.factorize(a.week.values)[0]; W = wk.max() + 1; rng = np.random.default_rng(seed)
    ba = [binned(y, x.p.values, wk, np.unique(np.quantile(x.p.values, np.linspace(0, 1, BINS + 1)[1:-1])), W) for x in (a, b)]
    full = [ap_from(P.sum(0), N.sum(0)) for P, N in ba]; chk = max(abs(full[0] - auprc(y, a.p.values)), abs(full[1] - auprc(y, b.p.values))); d = []
    for _ in range(draws):
        m = np.bincount(rng.integers(0, W, W), minlength=W).astype(float); d.append(ap_from(m @ ba[0][0], m @ ba[0][1]) - ap_from(m @ ba[1][0], m @ ba[1][1]))
    return float(full[0] - full[1]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(chk)


def main():
    runs = load(); rows, states = [], []; val = {}
    for m in ([] if A.paired_only else runs):
        for sp in A.splits:
            f = f"{m['dir']}/predictions/{sp}.parquet"
            if not os.path.exists(f): continue
            d = pd.read_parquet(f); y = d.y.values.astype(int); p = d.p.values.astype(float); day = pd.factorize(d.label_date.values)[0]
            if sp == "validation": val[m["run_id"]] = (y, p)
            q = None
            if sp != "validation" and m["run_id"] in val and not m.get("no_training"): q = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val[m["run_id"]][1], val[m["run_id"]][0]).predict(p)
            for sn, fn in STRATA.items():
                k = fn(d); rows.append(dict(run_id=m["run_id"], group=m["group"], grid=m["grid"], model=m["model"], variant=m.get("variant"), seed=m.get("seed"), split=sp, stratum=sn, **one(y[k], p[k], None if q is None else q[k], day[k])))
            if sp != "validation" or True:
                for s, g in d.groupby("state", observed=True): states.append(dict(run_id=m["run_id"], group=m["group"], split=sp, state=str(s), **{k: v for k, v in one(g.y.values.astype(int), g.p.values.astype(float), None, pd.factorize(g.label_date.values)[0]).items() if k in ("n", "pos", "auprc", "auroc")}))
        print("evaluated", m["run_id"], flush=True)
    os.makedirs(A.res, exist_ok=True)
    if A.paired_only:
        R = pd.read_csv(f"{A.res}/all_runs.csv")
    else:
        R = pd.DataFrame(rows); R.to_csv(f"{A.res}/all_runs.csv", index=False); pd.DataFrame(states).to_csv(f"{A.res}/per_state.csv", index=False)
    num = [c for c in R.columns if c in ("auprc", "auroc", "brier", "ece", "brier_iso", "ece_iso") or c.startswith("top")]
    if not A.paired_only: S = R[~R.grid].groupby(["group", "model", "split", "stratum"]).agg(**{"seeds": ("seed", "nunique"), "n": ("n", "first"), "pos": ("pos", "first"), **{f"{c}_{s}": (c, s) for c in num for s in ("mean", "std", "min", "max")}}).reset_index(); S.to_csv(f"{A.res}/seeds.csv", index=False)
    # paired comparisons on the test year, seed-averaged predictions
    def avg(g, sp):
        fs = [pd.read_parquet(f"{m['dir']}/predictions/{sp}.parquet") for m in runs if m["group"] == g and not m["grid"] and os.path.exists(f"{m['dir']}/predictions/{sp}.parquet")]
        if not fs: return None
        d = fs[0].copy(); d["p"] = np.mean([f.p.values for f in fs], 0); d["week"] = pd.to_datetime(d.label_date).dt.strftime("%G-%V"); return d
    pairs = [(a, b) for a, b in A.pairs]; out = []
    for sp in [s for s in A.splits if s != "validation"]:
        cache = {}
        for a, b in pairs:
            for g in (a, b): cache.setdefault(g, avg(g, sp))
            if cache[a] is None or cache[b] is None: continue
            A_, B_ = cache[a], cache[b]
            if not (len(A_) == len(B_) and (A_.cell_id.values == B_.cell_id.values).all() and (A_.label_date.values == B_.label_date.values).all()):
                # a model trained on another label (or with another valid-row rule) has other rows: compare on the cells both scored, against the labels of the first model
                j = A_.merge(B_[["cell_id", "label_date", "p"]], on=["cell_id", "label_date"], suffixes=("", "_b")); A_ = j.drop(columns="p_b"); B_ = j.assign(p=j.p_b).drop(columns="p_b")
                print("common rows", a, b, len(A_), "of", len(cache[a]), len(cache[b]), flush=True)
            dif, lo, hi, chk = paired(A_, B_); out.append(dict(split=sp, a=a, b=b, diff=dif, low=lo, high=hi, binning_check=chk, verdict="better" if lo > 0 else ("matches" if lo >= -0.005 else ("worse" if hi < 0 else "no clear difference"))))
            print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(f"{A.res}/paired.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--res", required=True); ap.add_argument("--splits", default="validation,test,blind"); ap.add_argument("--paired-only", action="store_true", help="only the paired comparisons; all_runs.csv, per_state.csv and seeds.csv in --res are kept")
    ap.add_argument("--pairs", default="c1_A:convlstm_A,c2_A:convlstm_A,convlstm_A:xgboost_nb_A,cnn_A:xgboost_nb_A,lstm_A:xgboost_nb_A,mlp_A:xgboost_nb_A,rf_nb_A:xgboost_nb_A,xgboost_nb_A:ref_fwi,xgboost_nb_A:ref_climatology,xgboost_nb_A:ref_persistence,xgboost_nb_A:xgboost_only_fire_history")
    A = ap.parse_args(); A.splits = A.splits.split(","); A.pairs = [p.split(":") for p in A.pairs.split(",")]; main()
