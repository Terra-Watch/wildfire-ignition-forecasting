"""Null-model sanity checks on the holdout: shuffled labels must not produce skill."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import calibration, harness, metrics
from wildfire.guards import provenance, sanity, shuffle
from wildfire.models.fit import Model
from wildfire.pipelines import features as feature_pipeline
from wildfire.pipelines.fit import arm_rows


def run(log=print) -> dict:
    provenance.require_clean()
    harness.check_split("holdout_year")
    s = config.load("evaluation")["sanity"]
    splits = config.load("evaluation")["splits"]
    table, columns = feature_pipeline.load_model_table()
    arm = "logistic_with_burn"
    cols = columns[arm]
    clim = columns["climatology_cell"][0]
    r = arm_rows(table, cols)
    tr = (table.year <= splits["fit_last_year"]).values & r
    ho = (table.year == splits["holdout_year"]).values & r
    yho = table.label.values[ho]
    prev = float(yho.mean())
    xtr, xho, y = table[cols].values[tr], table[cols].values[ho], table.label.values[tr]
    cell, week = table.cell.values[tr].astype(np.int64), table.week.values[tr].astype(np.int64)
    clim_v = table[clim].values
    ok_b = (table.year <= splits["fit_last_year"]).values & ~np.isnan(clim_v)
    hb = ho & ~np.isnan(clim_v)
    pb, _ = calibration.calibrate_climatology(clim_v[ok_b], table.label.values[ok_b], table.year.values[ok_b], clim_v[hb])
    yb = table.label.values[hb]
    ap_clim = metrics.average_precision(yb, clim_v[hb])
    in_b = ~np.isnan(clim_v[ho])

    def fit_score(ys):
        return Model("logistic", s["shuffle_fit_params"]).fit(xtr, ys).predict(xho)

    ap_a = []
    for sd in range(s["global_shuffle_seeds"]):
        ys = np.random.default_rng(s["global_shuffle_seed_base"] + sd).permutation(y)
        assert ys.sum() == y.sum()
        ap_a.append(metrics.average_precision(yho, fit_score(ys)))
        log("global shuffle seed", sd, round(ap_a[-1], 5))
    ap_b = []
    for sd in range(s["cell_week_shuffle_seeds"]):
        ys = shuffle.shuffle_within(y, cell * 100 + week, s["cell_week_shuffle_seed_base"] + sd)
        assert ys.sum() == y.sum()
        p = fit_score(ys)
        ap_b.append(metrics.average_precision(yho[in_b], p[in_b]))
        log("cell-week shuffle seed", sd, round(ap_b[-1], 5), "climatology", round(ap_clim, 5))
    rng = np.random.default_rng(s["permuted_holdout_seed"])
    ap_c = [metrics.average_precision(rng.permutation(yho), rng.random(len(yho))) for _ in range(s["permuted_holdout_repeats"])]
    decisions = sanity.sanity_decisions(ap_a, prev, ap_b, ap_clim, ap_c)
    screen = sanity.univariate_screen(xtr, y, cols)
    out = config.output_dir("results")
    summary = pd.DataFrame([
        dict(check="global label shuffle", mean_ap=float(np.mean(ap_a)), prevalence=prev, passed=decisions["a"]),
        dict(check="cell x ISO-week shuffle", max_ap=float(np.max(ap_b)), climatology_ap=ap_clim, passed=decisions["b"]),
        dict(check="permuted holdout labels", mean_ap=float(np.mean(ap_c)), prevalence=prev, passed=decisions["c"]),
    ])
    provenance.stamp_df(summary, out / "sanity.csv")
    provenance.stamp_df(screen, out / "sanity_univariate.csv")
    log("decisions", decisions)
    return dict(decisions=decisions, summary=summary, screen=screen, ap_global=ap_a, ap_cell_week=ap_b, ap_permuted=ap_c)
