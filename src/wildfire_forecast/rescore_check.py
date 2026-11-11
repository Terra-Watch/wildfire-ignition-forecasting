"""Checkpoint re-score check: reload a saved model file, predict the validation year again and compare with the saved predictions of the same run.
Tabular models are expected to match to floating-point noise; neural models were scored in bfloat16 on a GPU and are re-scored here in float32 on a CPU, so they are expected to match only closely
(reported: max absolute difference, AUPRC of both). usage: WILDFIRE_DATA=... python scripts/11_rescore_check.py --runs DIR --ids xgboost_A_s0,lr_A_s0 --out results_v4/rescore_check.csv"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as auprc
from wildfire_forecast import harness as H
from wildfire_forecast.score_blind import load_predictor

ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True); ap.add_argument("--rows", type=int, default=300000); a = ap.parse_args()
day = lambda s: (pd.Timestamp(s) - pd.Timestamp("2019-01-01")).days; out = []
for rid in a.ids.split(","):
    d = f"{a.runs}/{rid}"; man = json.load(open(f"{d}/manifest.json")); saved = pd.read_parquet(f"{d}/predictions/validation.parquet")
    def cube_for(channels, torch_, dev="cpu"): return H.Cube(H.DATASET, channels, dev, t_range=(day("2023-12-20"), day("2025-01-05")), use_torch=torch_)
    cube, pr = load_predictor(d, man, cube_for); rs = H.pick_rows(cube, None, "validation", man.get("label", "y_forest"))
    assert (rs.cell_idx.values == saved.cell_idx.values).all() and (rs.label_date.values == saved.label_date.values).all(), "row order differs"
    idx = np.sort(np.random.default_rng(0).choice(len(rs), min(a.rows, len(rs)), replace=False)); p = np.asarray(pr(rs.iloc[idx].reset_index(drop=True)), float); q = saved.p.values[idx].astype(float)
    out.append(dict(run=rid, model=man["model"], rows=len(idx), max_abs_diff=float(np.abs(p - q).max()), mean_abs_diff=float(np.abs(p - q).mean()), auprc_saved=float(auprc(saved.y.values[idx], q)), auprc_rescored=float(auprc(saved.y.values[idx], p))))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(a.out, index=False)
