"""More paired comparisons with the same method as evaluate.py (seed-averaged predictions, week-block bootstrap), appended to a paired.csv.
usage: python src/paired_extra.py --runs DIR --pairs c2_A:xgboost_A,c1_A:xgboost_A --splits test,blind --csv results_v4/paired.csv"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import paired

ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--splits", default="test,blind"); ap.add_argument("--csv", required=True); a = ap.parse_args()
def avg(g, sp):
    fs = sorted(glob.glob(f"{a.runs}/{g}_s[0-9]/predictions/{sp}.parquet")) + sorted(glob.glob(f"{a.runs}/{g}/predictions/{sp}.parquet")); d = pd.read_parquet(fs[0]).copy(); d["p"] = np.mean([pd.read_parquet(f, columns=["p"]).p.values for f in fs], 0); d["week"] = pd.to_datetime(d.label_date).dt.strftime("%G-%V"); return d
rows = []
for sp in a.splits.split(","):
    for pr in a.pairs.split(","):
        x, y = pr.split(":"); A, B = avg(x, sp), avg(y, sp); assert (A.cell_id.values == B.cell_id.values).all() and (A.label_date.values == B.label_date.values).all()
        dif, lo, hi, chk = paired(A, B); rows.append(dict(split=sp, a=x, b=y, diff=dif, low=lo, high=hi, binning_check=chk, verdict="better" if lo > 0 else ("matches" if lo >= -0.005 else ("worse" if hi < 0 else "no clear difference")))); print(rows[-1], flush=True)
old = pd.read_csv(a.csv); new = pd.DataFrame(rows); out = pd.concat([old[~old.set_index(["split", "a", "b"]).index.isin(new.set_index(["split", "a", "b"]).index)], new]); out.to_csv(a.csv, index=False)
