"""Which block size should the forecast be shown at: the 0.1 degree cell (about 11 km), 0.25 degrees (about 28 km) or 0.5 degrees (about 55 km)?

Same predictions for all three: the five-seed XGBoost mean on the test year 2025, calibrated per cell on 2024, then aggregated to blocks (chance that at least one cell of the
block has a detection) and calibrated again per block size on 2024. Three things are compared on 2025:
  1. calibration (expected calibration error) and discrimination (average precision, and lift over the base rate, because bigger blocks have a higher base rate by construction);
  2. coverage at an equal patrol budget: if a division can watch only K forest cells a day, how many of the day's detections fall in the blocks it picks, and what share of the cells it
     watches actually had a detection. Equal watched cells, not equal numbers of blocks, so the scales are comparable;
  3. the same for two ways of ranking blocks: by the block's chance (what the levels use) and by density (mean cell chance in the block, the efficiency per cell watched).
Usage: python scripts/18_block_scale_study.py --runs runs --cells ../terrawatch-datasets/dataset/forest_all_gfs/cells.parquet --out results/block_scale_study
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--cells", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
SCALES = (0.1, 0.25, 0.5); BUDGETS = (100, 300, 1000); ACTIVE_MIN = 50      # budgets in forest cells watched per day; an active day has at least this many detection cells


def load(split):
    parts = [pd.read_parquet(f"{a.runs}/xgboost_A_s{s}/predictions/{split}.parquet", columns=["cell_idx", "cell_id", "date", "label_date", "y", "p"]) for s in range(a.seeds)]
    d = parts[0].copy(); d["p"] = np.mean([x["p"].to_numpy() for x in parts], axis=0)
    return d


cells = pd.read_parquet(a.cells, columns=["cell_id", "lat", "lon"]).drop_duplicates("cell_id")
val, test = load("validation"), load("test")
for d in (val, test): d[["lat", "lon"]] = d[["cell_id"]].merge(cells, on="cell_id", how="left")[["lat", "lon"]].to_numpy()
test["state"] = pd.read_parquet(f"{a.runs}/xgboost_A_s0/predictions/test.parquet", columns=["state"]).state.astype(str).to_numpy(); val["state"] = "x"
cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val.p.to_numpy(), val.y.to_numpy())
for d in (val, test): d["q"] = cal.predict(d.p.to_numpy()).clip(1e-6, 1 - 1e-6)           # calibrated cell chance


def blocks(d, s):
    g = d.assign(bl=np.floor(d.lat / s + 1e-9).astype(int) * 100000 + np.floor(d.lon / s + 1e-9).astype(int), lq=np.log1p(-d.q))
    out = g.groupby(["label_date", "bl"], sort=False).agg(n=("y", "size"), y=("y", "sum"), lq=("lq", "sum"), q=("q", "mean"), st=("state", "first")).reset_index()
    out["chance"] = 1 - np.exp(out.lq); out["hit"] = (out.y > 0).astype(int)
    return out


def ece(pred, y, bins=20):
    o = np.argsort(pred); chunks = np.array_split(o, bins)
    return float(sum(len(c) * abs(pred[c].mean() - y[c].mean()) for c in chunks) / len(pred))


def coverage_by_state(b, rank, budget):
    """The same, but every state gets its share of the budget (in proportion to the cells it has that day), as each forest department patrols its own land."""
    rec, prec = [], []
    for _, g in b.groupby("label_date", sort=False):
        tot = g.y.sum()
        if tot < ACTIVE_MIN: continue
        total_n, hit, watched = g.n.sum(), 0, 0
        for _, h in g.groupby("st"):
            k = round(budget * h.n.sum() / total_n); h = h.sort_values(rank, ascending=False); sel = h[h.n.cumsum() <= k]; hit += sel.y.sum(); watched += sel.n.sum()
        rec.append(hit / tot); prec.append(hit / max(watched, 1))
    return float(np.mean(rec)), float(np.mean(prec))


def coverage_months(b, rank, budget):
    out = {}
    for m, g in b.assign(m=b.label_date.dt.month).groupby("m"):
        out[int(m)] = coverage(g, rank, budget)[0]
    return out


def daily_recall(b, rank, budget):
    """Recall per active day, so differences between scales can be given an interval by resampling days."""
    out = {}
    for day, g in b.groupby("label_date", sort=False):
        tot = g.y.sum()
        if tot < ACTIVE_MIN: continue
        g = g.sort_values(rank, ascending=False); out[day] = g[g.n.cumsum() <= budget].y.sum() / tot
    return pd.Series(out)


def coverage(b, rank, budget):
    """Per day: take blocks in rank order while they fit in `budget` watched cells (a block is taken whole, so a coarse scale cannot fill the budget exactly); recall of detection cells, share of watched cells with a detection."""
    rec, prec, days = [], [], 0
    for _, g in b.groupby("label_date", sort=False):
        tot = g.y.sum()
        if tot < ACTIVE_MIN: continue
        g = g.sort_values(rank, ascending=False); sel = g[g.n.cumsum() <= budget]      # whole blocks only, never more than the budget
        rec.append(sel.y.sum() / tot); prec.append(sel.y.sum() / max(sel.n.sum(), 1)); days += 1
    return float(np.mean(rec)), float(np.mean(prec)), days


rows = []
for s in SCALES:
    bv, bt = blocks(val, s), blocks(test, s)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(bv.chance.to_numpy(), bv.hit.to_numpy())
    bt["pred"] = iso.predict(bt.chance.to_numpy()); base = bt.hit.mean()
    r = {"block_deg": s, "approx_km": round(s * 111), "blocks_per_day": round(len(bt) / bt.label_date.nunique()), "cells_per_block": round(bt.n.mean(), 1), "base_rate": round(base, 4),
         "ap": round(average_precision_score(bt.hit, bt.pred), 4), "ece": round(ece(bt.pred.to_numpy(), bt.hit.to_numpy()), 4)}
    r["lift"] = round(r["ap"] / base, 2)
    for rank in ("chance", "q"):
        for K in BUDGETS:
            rec, prec, days = coverage(bt, rank, K); name = "chance" if rank == "chance" else "density"
            r[f"recall@{K}_{name}"] = round(rec, 3); r[f"precision@{K}_{name}"] = round(prec, 3)
    for K in BUDGETS:
        rec, prec = coverage_by_state(bt, "q", K); r[f"recall@{K}_per_state_density"] = round(rec, 3); r[f"precision@{K}_per_state_density"] = round(prec, 3)
    mm = coverage_months(bt, "q", 300); r["recall@300_march"] = round(mm.get(3, float("nan")), 3); r["recall@300_feb"] = round(mm.get(2, float("nan")), 3); r["recall@300_apr"] = round(mm.get(4, float("nan")), 3)
    r["active_days"] = days; rows.append(r)
rng = np.random.default_rng(0); ci = []
daily = {s_: {K: daily_recall(blocks(test, s_), "q", K) for K in BUDGETS} for s_ in SCALES}
for K in BUDGETS:
    for coarse in (0.25, 0.5):
        dd = (daily[0.1][K] - daily[coarse][K]).dropna(); d = dd.to_numpy()
        wk = dd.groupby(pd.Index(dd.index).isocalendar().week.to_numpy()).apply(lambda x: x.to_numpy()).tolist()     # resample whole ISO weeks: consecutive days are not independent
        boot = [np.concatenate([wk[i] for i in rng.integers(0, len(wk), len(wk))]).mean() for _ in range(2000)]
        ci.append({"budget_cells": K, "cell_minus_block_deg": coarse, "mean_diff_recall": round(float(d.mean()), 3), "ci95_low": round(float(np.percentile(boot, 2.5)), 3), "ci95_high": round(float(np.percentile(boot, 97.5)), 3), "days_cell_better": int((d > 0).sum()), "days": len(d), "weeks": len(wk)})
pd.DataFrame(ci).to_csv(a.out + "_diffs.csv", index=False); print(pd.DataFrame(ci).to_string())
res = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
res.to_csv(a.out + ".csv", index=False); json.dump(res.to_dict("records"), open(a.out + ".json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); print(res.T.to_string())
