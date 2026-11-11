"""95 per cent paired-free intervals for the AUPRC of single models: calendar-week block bootstrap (2,000 draws) of the seed-averaged predictions, on the binned counts used by evaluate.py.
usage: python scripts/06_confidence_intervals.py --runs DIR --out results_v4/intervals.csv --splits test,blind [--groups xgboost_A,...]"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from wildfire_forecast.evaluate import binned, ap_from, BINS

ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--splits", default="test,blind")
ap.add_argument("--groups", default="xgboost_A,lstm_A,rf_A,c2_A,c1_A,convlstm_A,mlp_A,cnn_A,lr_A,convlstm_cell_A,c2_cell_A,xgboost_nb_A,rf_nb_A,ref_climatology,ref_persistence,ref_fwi"); a = ap.parse_args()
rows = []
for sp in a.splits.split(","):
    for g in a.groups.split(","):
        fs = [pd.read_parquet(f) for f in sorted(glob.glob(f"{a.runs}/{g}_s[0-9]/predictions/{sp}.parquet")) + sorted(glob.glob(f"{a.runs}/{g}/predictions/{sp}.parquet"))]
        if not fs: continue
        d = fs[0][["label_date", "y"]].copy(); d["p"] = np.mean([f.p.values for f in fs], 0); y = d.y.values; wk = pd.factorize(pd.to_datetime(d.label_date).dt.strftime("%G-%V"))[0]; W = wk.max() + 1
        P, N = binned(y, d.p.values, wk, np.unique(np.quantile(d.p.values, np.linspace(0, 1, BINS + 1)[1:-1])), W); full = ap_from(P.sum(0), N.sum(0)); rng = np.random.default_rng(0); b = []
        for _ in range(2000):
            m = np.bincount(rng.integers(0, W, W), minlength=W).astype(float); b.append(ap_from(m @ P, m @ N))
        rows.append(dict(split=sp, group=g, seeds=len(fs), auprc_seed_mean_pred=full, low=float(np.percentile(b, 2.5)), high=float(np.percentile(b, 97.5)))); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
