"""Leave-year-out: fit on the years before the test year, score all rows of the test year."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import bootstrap, harness
from wildfire.evaluation.views import views
from wildfire.guards import provenance
from wildfire.models import arms
from wildfire.pipelines import features as feature_pipeline
from wildfire.pipelines.fit import arm_rows, final_fit


def run(log=print) -> dict:
    provenance.require_clean()
    harness.check_split("leave_year_out")
    t0 = time.time()
    splits = config.load("evaluation")["splits"]["leave_year_out"]
    caps = config.load("models")["fit_row_caps"]
    table, columns = feature_pipeline.load_model_table()
    fitted, ref = arms.fitted_arms(), arms.reference_arm()
    fit = (table.year <= splits["fit_last_year"]).values
    test = (table.year == splits["test_year"]).values
    scores = {}
    for a in fitted:
        cols = columns[a]
        mod = final_fit(table, a, columns, fit, cap=caps["leave_year_out"], thin_seed=caps["leave_year_out_thin_seed"])
        s = np.full(int(test.sum()), np.nan)
        idx = np.flatnonzero(arm_rows(table, cols)[test])
        s[idx] = mod.predict(table[cols].values[test & arm_rows(table, cols)])
        scores[a] = s
        log("leave-year-out", a)
    df = table[test].reset_index(drop=True)
    scores[ref] = df[columns[ref][0]].values
    rows = []
    for a, s in scores.items():
        ok = ~np.isnan(s)
        rows.append(dict(arm=a, rows=int(ok.sum()), **views(df[ok], s[ok])))
    table_out = pd.DataFrame(rows)
    out = config.output_dir("results")
    provenance.stamp_df(table_out, out / "leave_year_out_views.csv")

    by_state = {}
    ok_all = np.ones(len(df), bool)
    for s in scores.values():
        ok_all &= ~np.isnan(s)
    for st, g in df[ok_all].groupby("state"):
        ix = g.index.values
        dd = g.assign(cluster=g.state + "|" + g.year.astype(int).astype(str) + "|" + g.week.astype(int).astype(str).str.zfill(2)).reset_index(drop=True)
        by_state[st] = (dd, {a: scores[a][ix] for a in scores})
    draws = config.load("evaluation")["bootstrap"]["leave_year_out_draws"]
    paired = {}
    for reference, names in ((ref, [ref] + fitted), ("logistic_weather_static", ["logistic_with_burn"])):
        names = [n for n in names if n != reference]
        paired[reference] = bootstrap.paired_cluster_bootstrap(by_state, names, reference, nboot=draws)
    log("done", round(time.time() - t0))
    return dict(views=table_out, paired=paired, scores=scores)
