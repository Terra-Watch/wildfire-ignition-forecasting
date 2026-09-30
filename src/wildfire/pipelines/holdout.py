"""Score the holdout year: fit on earlier years, compare every arm with the climatology reference."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import calibration, harness, prior_shift
from wildfire.guards import provenance
from wildfire.labels import weights
from wildfire.models import arms
from wildfire.pipelines import features as feature_pipeline
from wildfire.pipelines.fit import arm_rows, final_fit


def run(log=print) -> dict:
    provenance.require_clean()
    harness.check_split("holdout_year")
    t0 = time.time()
    splits = config.load("evaluation")["splits"]
    table, columns = feature_pipeline.load_model_table()
    fitted, ref = arms.fitted_arms(), arms.reference_arm()
    clim_col = columns[ref][0]
    fit_rows = (table.year <= splits["fit_last_year"]).values
    ho = (table.year == splits["holdout_year"]).values
    ok_ref = fit_rows & ~np.isnan(table[clim_col].values)
    iso = calibration.Isotonic().fit(table[clim_col].values[ok_ref], table.label.values[ok_ref], table.year.values[ok_ref])
    pi_s = {ref: float(table.label.values[ok_ref].mean())}
    common = ho.copy()
    for a in fitted:
        common &= arm_rows(table, columns[a])
    H = table[common].reset_index(drop=True)
    y = H.label.values
    probs = {ref: iso.transform(H[clim_col].values)}
    scores = {ref: H[clim_col].values}
    models = {}
    for a in fitted:
        t = time.time()
        models[a] = final_fit(table, a, columns, fit_rows)
        p = models[a].predict(H[columns[a]].values)
        probs[a] = scores[a] = p
        pi_s[a] = models[a].fit_prevalence
        log("fitted", a, "fit prevalence", round(pi_s[a], 5), "s", round(time.time() - t))
    out = config.output_dir("results")
    pd.DataFrame({"cell": H.cell.values, "day": H.day.values, "label": y, **{f"p_{a}": probs[a] for a in probs}}) \
        .to_pickle(out / "holdout_predictions.pkl")

    rate = weights.base_rates(table)
    pit = np.array([rate[s] for s in H.state.values])
    masked = np.unique(table.cell.values[table.t2m.isna().values])
    frame = weights.frame_sizes([splits["holdout_year"]], exclude_cells=masked)
    w = prior_shift.design_weights(y, weights.strata(H), frame)
    corr = {a: prior_shift.prior_shift_probs(probs[a], pit, pi_s[a]) for a in probs}
    clusters = weights.iso_clusters(H)
    summary = dict(rows=len(H), onsets=int(y.sum()), prevalence=float(y.mean()), clusters=len(np.unique(clusters)),
                   masked_cells=len(masked), base_rates=rate, fit_prevalence=pi_s)
    log(summary)

    def compare(names, reference, label):
        before = harness.score_arms(H, {a: scores[a] for a in names}, {a: probs[a] for a in names}, reference, clusters)
        after = harness.score_arms(H, {a: scores[a] for a in names}, {a: corr[a] for a in names}, reference, clusters, weights=w)
        tb = harness.result_table(before, names, reference)
        ta = harness.result_table(after, names, reference)
        provenance.stamp_df(tb, out / f"holdout_{label}_before_prior_shift.csv")
        provenance.stamp_df(ta, out / f"holdout_{label}_after_prior_shift.csv")
        provenance.stamp_df(before["_views"], out / f"holdout_{label}_views.csv")
        return before, after, tb, ta

    results = {}
    results["headline"] = compare(["logistic_weather_static", "logistic_with_burn"], "logistic_weather_static", "headline")
    results["all_arms"] = compare([ref] + fitted, ref, "all_arms")
    for a in fitted:   # arms whose row set differs from the common set are also reported on their own rows
        r = arm_rows(table, columns[a]) & ho
        if r.sum() != common.sum():
            Hs = table[r].reset_index(drop=True)
            ps = {ref: iso.transform(Hs[clim_col].values), a: models[a].predict(Hs[columns[a]].values)}
            ob = harness.score_arms(Hs, {ref: Hs[clim_col].values, a: ps[a]}, ps, ref, weights.iso_clusters(Hs))
            provenance.stamp_df(harness.result_table(ob, [a], ref), out / f"holdout_own_rows_{a}.csv")
    log("done", round(time.time() - t0))
    return dict(summary=summary, **results)
