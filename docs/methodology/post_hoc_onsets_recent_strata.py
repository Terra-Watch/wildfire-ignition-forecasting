"""Post hoc diagnostic (docs/decisions/2026-10-01-onsets-recent-strata.md): validation rows split by whether the neighbourhood block's
recent-neighbour-onset count is zero or above zero, for boosting@tabular and boosting@enriched. Read-only: it reads the saved
rolling-origin predictions and the neighbourhood array; no model is refitted, nothing in the baseline is changed.

Run from the repository root with the .env variables set:  python docs/methodology/post_hoc_onsets_recent_strata.py"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from wildfire import config
from wildfire.evaluation import bootstrap, ladder, metrics, verdict
from wildfire.features import neighbourhood
from wildfire.pipelines import baseline_tables, dataset, rolling_origin

ARMS = ("boosting@tabular", "boosting@enriched")
REFERENCE = "climatology_cell"                                  # reported beside them as the baseline arm
FEATURES = ("ring_8_onsets_recent", "ring_24_onsets_recent")     # primary: the 8 nearest neighbours; secondary: the 24-cell ring
OUT = config.output_dir("results") / "post_hoc_onsets_recent_strata.csv"


def stratum_values() -> dict:
    """{feature: onsets_recent of every validation row}, in the row order of the pooled predictions."""
    ds = dataset.load_dataset()
    names = neighbourhood.column_names()
    out = {f: [] for f in FEATURES}
    for year in config.load("baseline")["validation"]["validation_years"]:
        val = dataset.validation_rows(ds, year)
        frame, _ = rolling_origin.load_year(year)
        assert len(frame) == len(val) and (frame.cell.values == ds.table.cell.values[val]).all() and (frame.day.values == ds.table.day.values[val]).all()
        block = np.asarray(ds.arrays["neighbourhood"][val])
        for f in FEATURES:
            out[f].append(block[:, names.index(f)])
    return {f: np.concatenate(v) for f, v in out.items()}


def stats(y, s) -> dict:
    return dict(rows=len(y), onsets=int(y.sum()), prevalence=float(y.mean()) if len(y) else np.nan,
                auc=float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
                ap=float(metrics.average_precision(y, s)) if y.sum() > 0 else np.nan)


def main() -> None:
    rows, corrected, scores = baseline_tables.load_pooled()
    values = stratum_values()
    y = rows.label.values
    clusters = ladder.cluster_labels(rows)
    out = []
    for feature, v in values.items():
        assert not np.isnan(v).any()
        for period in ["pooled"] + sorted(rows.year.unique().tolist()):
            in_period = np.ones(len(rows), bool) if period == "pooled" else (rows.year.values == period)
            for name, stratum in (("zero", v == 0), ("above_zero", v > 0)):
                m = in_period & stratum
                for arm in (REFERENCE,) + ARMS:
                    out.append(dict(feature=feature, period=period, stratum=name, arm=arm, **stats(y[m], scores[arm][m]),
                                    share_of_positives=float(y[m].sum() / y[in_period].sum()), share_of_rows=float(m.sum() / in_period.sum())))
    table = pd.DataFrame(out)
    # the enriched gain inside each stratum (pooled): paired ISO-week cluster bootstrap of the AP difference, same code as the contrasts
    gains = []
    for feature, v in values.items():
        for name, stratum in (("zero", v == 0), ("above_zero", v > 0)):
            sub = rows[stratum].reset_index(drop=True)
            arms = {a: dict(score=scores[a][stratum], prob=corrected[a][stratum]) for a in ARMS}
            o = bootstrap.paired_week_blocked_bootstrap(sub.label.values, ladder.cluster_labels(sub), arms, ARMS[0], weights=sub.weight.values)
            d = o[ARMS[1]]["delta_ap"]
            a0, a1 = [stats(sub.label.values, scores[a][stratum]) for a in ARMS]
            gains.append(dict(feature=feature, stratum=name, rows=len(sub), onsets=int(sub.label.sum()), ap_tabular=a0["ap"], ap_enriched=a1["ap"],
                              delta_ap=d["point"], ap_lo99=d["lo99"], ap_hi99=d["hi99"], ap_se=d["se"], mde=verdict.minimum_detectable_effect(d["se"]),
                              random_baseline_ap=float(sub.label.mean()), ap_climatology=stats(sub.label.values, scores[REFERENCE][stratum])["ap"], delta_auc=a1["auc"] - a0["auc"]))
    gains = pd.DataFrame(gains)
    table.to_csv(OUT, index=False)
    gains.to_csv(OUT.with_name("post_hoc_onsets_recent_gain.csv"), index=False)
    pd.set_option("display.width", 250)
    print(table[table.period == "pooled"].round(4).to_string())
    print(gains.round(4).to_string())


if __name__ == "__main__":
    main()
