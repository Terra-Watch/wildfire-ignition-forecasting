"""Test-year (2025) AUPRC of the main models by land-cover class of the cell: crop-dominated (WorldCover crop share >= 0.5), forest-dominated (forest-land share >= 0.5 and not crop-dominated), other.
Seed means over the saved test predictions. usage: python src/landcover_split.py --runs DIR --out results_v4/landcover_split.csv [--split test]"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as auprc, roc_auc_score as auroc

ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--split", default="test"); a = ap.parse_args()
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "wildfire-dataset", "dataset", "forest_unseen_gfs" if a.split == "blind" else "forest_all_gfs")      # the blind states have their own cells
c = pd.read_parquet(f"{D}/cells.parquet"); cls = np.where(c.crop >= 0.5, "crop_dominated", np.where(c.forest_land_frac >= 0.5, "forest_dominated", "other"))
GROUPS = ["xgboost_A", "lstm_A", "rf_A", "c2_A", "c1_A", "convlstm_A", "mlp_A", "cnn_A", "lr_A", "convlstm_cell_A", "c2_cell_A", "ref_climatology", "ref_persistence", "ref_fwi"]; rows = []
for g in GROUPS:
    for d in sorted(glob.glob(f"{a.runs}/{g}") + glob.glob(f"{a.runs}/{g}_s[0-9]")):
        f = f"{d}/predictions/{a.split}.parquet"
        if not os.path.exists(f): continue
        x = pd.read_parquet(f, columns=["cell_idx", "y", "p"]); k = cls[x.cell_idx.values]
        for name in ("crop_dominated", "forest_dominated", "other"):
            m = k == name; rows.append(dict(group=g, run=os.path.basename(d), landcover=name, cells=int((cls == name).sum()), n=int(m.sum()), pos=int(x.y[m].sum()), auprc=auprc(x.y[m], x.p[m]), auroc=auroc(x.y[m], x.p[m])))
R = pd.DataFrame(rows); S = R.groupby(["group", "landcover"]).agg(cells=("cells", "first"), n=("n", "first"), pos=("pos", "first"), auprc=("auprc", "mean"), auroc=("auroc", "mean")).reset_index(); S.to_csv(a.out, index=False)
print(S.pivot(index="group", columns="landcover", values="auprc").round(3).to_string()); print(S[S.group == "xgboost_A"][["landcover", "cells", "pos"]].to_string(index=False))
