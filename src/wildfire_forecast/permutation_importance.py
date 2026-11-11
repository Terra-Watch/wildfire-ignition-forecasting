"""Permutation importance on the validation year for a saved XGBoost or random-forest run: for every input channel and for every input group (the groups of the ablation table),
shuffle those columns across rows (five repeats) and record the fall in AUPRC. Main-model setting (single-cell features, no neighbour features).
usage: WILDFIRE_DATA=... python scripts/08_permutation_importance.py --model xgboost --path model.json --out permutation_xgboost.csv [--rows 400000] [--repeats 5]"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as auprc
from wildfire_forecast import harness as H

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["xgboost", "rf"]); ap.add_argument("--path", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rows", type=int, default=400000); ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
names = [str(n) for n in np.load(f"{H.DATA}/scaler_forest_AG_train.npz", allow_pickle=True)["names"]]
day = lambda s: (pd.Timestamp(s) - pd.Timestamp("2019-01-01")).days
cube = H.Cube(H.DATASET, names, "cpu", t_range=(day("2023-12-20"), day("2025-01-05")), use_torch=False)
rs = H.pick_rows(cube, None, "validation", "y_forest"); rng = np.random.default_rng(a.seed); rs = rs.iloc[np.sort(rng.choice(len(rs), min(a.rows, len(rs)), replace=False))].reset_index(drop=True)
X = H.tab_matrix(cube, rs, False); y = rs.y.values
if a.model == "xgboost":
    import xgboost as xgb
    b = xgb.Booster(); b.load_model(a.path); n = b.num_boosted_rounds(); predict = lambda M: b.predict(xgb.DMatrix(M), iteration_range=(0, n))
else:
    import joblib
    m = joblib.load(a.path); predict = lambda M: m.predict_proba(M)[:, 1]
base = auprc(y, predict(X)); print("baseline validation AUPRC (subsample)", round(base, 4), "rows", len(y), "positives", int(y.sum()), flush=True)
units = [(n, [names.index(n)]) for n in names] + [("GROUP:" + g, [names.index(c) for c in cs if c in names]) for g, cs in H.GROUPS.items()]
out = []
for label, cols in units:
    drops = []
    for r in range(a.repeats):
        P = X.copy(); perm = np.random.default_rng(a.seed * 100 + r).permutation(len(P)); P[:, cols] = X[perm][:, cols]; drops.append(base - auprc(y, predict(P)))
    out.append(dict(unit=label, columns=len(cols), drop_mean=float(np.mean(drops)), drop_sd=float(np.std(drops)))); print(label, round(out[-1]["drop_mean"], 4), flush=True)
pd.DataFrame(out).assign(model=a.model, baseline=base, rows=len(y), repeats=a.repeats).to_csv(a.out, index=False)
