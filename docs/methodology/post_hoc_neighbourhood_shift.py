"""Post hoc, descriptive: shift of the 12 neighbourhood-block columns between the rows `tabular_network@enriched` was standardised on and the
fresh AP and UK rows it scored. Read-only: it reads the model table, the enriched arrays and the saved fresh predictions; nothing is
refitted or re-scored. The standardisation is the network's own (`Standardiser`: mean and SD of the fit rows without the early-stopping
year, SD + epsilon, float32), reproduced for these columns, which are standardised independently of the others.
Run from the repository root with the .env variables set."""
import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import metrics
from wildfire.features import neighbourhood
from wildfire.pipelines import dataset, fresh_states

R = config.output_dir("results")
COLS = neighbourhood.column_names()
THRESHOLDS = (5, 10)                 # fixed before looking at the data: rows whose largest standardised value exceeds these


def main() -> None:
    ds = dataset.load_dataset()
    last_year = config.load("baseline")["periods"]["last_development_fold"]
    fit_idx = dataset.fit_rows(ds, last_year + 1)
    stop = ds.table.year.values[fit_idx] == last_year               # early-stopping year of the network, not used to standardise
    train = fit_idx[~stop]
    x = np.asarray(ds.arrays["neighbourhood"][train])
    eps = config.load("models")["logistic"]["scale_epsilon"]
    mu = x.mean(0, dtype=np.float64).astype(np.float32)
    sd = (x.std(0, dtype=np.float64) + eps).astype(np.float32)
    dv = dataset.load_dataset(fresh_states.TABLE, fresh_states.PREFIX)
    clim = ds.tabular_columns["climatology_cell"][0]
    val = np.flatnonzero(dv.complete & dv.table[clim].notna().values)       # the rows that were scored
    pred = pd.read_parquet(R / "fresh_predictions.parquet", columns=["cell", "day", "state", "label"])
    assert len(pred) == len(val) and (pred.cell.values == dv.table.cell.values[val]).all() and (pred.day.values == dv.table.day.values[val]).all()
    sets = {"fit (standardiser rows)": x}
    for st in ("AP", "UK"):
        sets[st] = np.asarray(dv.arrays["neighbourhood"][val[pred.state.values == st]])
    stat = []
    for name, a in sets.items():
        z = (a - mu) / sd
        for j, c in enumerate(COLS):
            stat.append(dict(rows=name, column=c, n=len(a), mean=float(a[:, j].mean(dtype=np.float64)), sd=float(a[:, j].std(dtype=np.float64)),
                             p99=float(np.percentile(a[:, j], 99)), fit_mean=float(mu[j]), fit_sd=float(sd[j]),
                             z_min=float(z[:, j].min()), z_max=float(z[:, j].max()), z_p99=float(np.percentile(z[:, j], 99))))
    stats = pd.DataFrame(stat)
    stats.to_csv(R / "post_hoc_neighbourhood_shift_columns.csv", index=False)
    # which rows carry large standardised values, and what the network scored on them (saved predictions only)
    groups = []
    for st in ("AP", "UK"):
        m = pred.state.values == st
        z = np.abs((sets[st] - mu) / sd)
        worst = z.max(1)
        sub = pred[m].reset_index(drop=True)
        p = pd.read_parquet(R / "fresh_predictions.parquet", columns=[c for c in pd.read_parquet(R / "fresh_predictions.parquet").columns if c.startswith("p_")])[m].reset_index(drop=True)
        seeds = {lvl: [c for c in p.columns if c.startswith(f"p_tabular_network@{lvl}#")] for lvl in ("tabular", "enriched")}
        for label, sel in ((f"max |z| <= {THRESHOLDS[0]}", worst <= THRESHOLDS[0]), (f"{THRESHOLDS[0]} < max |z| <= {THRESHOLDS[1]}", (worst > THRESHOLDS[0]) & (worst <= THRESHOLDS[1])),
                           (f"max |z| > {THRESHOLDS[1]}", worst > THRESHOLDS[1]), ("all rows", np.ones(len(sub), bool))):
            y = sub.label.values[sel]
            row = dict(state=st, group=label, rows=int(sel.sum()), share_of_rows=float(sel.mean()), onsets=int(y.sum()), prevalence=float(y.mean()) if len(y) else np.nan)
            if y.sum() > 0 and sel.sum() > 0:
                for lvl, cols in seeds.items():
                    row[f"network_{lvl}_AP_mean_of_seeds"] = float(np.mean([metrics.average_precision(y, p[c].values[sel]) for c in cols]))
                row["boosting_enriched_AP"] = float(metrics.average_precision(y, p["p_boosting@enriched"].values[sel]))
                row["boosting_tabular_AP"] = float(metrics.average_precision(y, p["p_boosting@tabular"].values[sel]))
            groups.append(row)
        top = np.unravel_index(np.argmax(z), z.shape)
        print(st, "largest standardised value", round(float(z.max()), 2), "column", COLS[top[1]], "signed", round(float(((sets[st] - mu) / sd)[top]), 2))
    pd.DataFrame(groups).to_csv(R / "post_hoc_neighbourhood_shift_groups.csv", index=False)
    pd.set_option("display.width", 250)
    print(stats[stats.rows != "fit (standardiser rows)"].round(3).to_string())
    print(pd.DataFrame(groups).round(4).to_string())


if __name__ == "__main__":
    main()
