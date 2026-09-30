"""Out-of-fold table: forward-chaining folds, one validation year each, January-July rows, common row set."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import calibration, harness, prior_shift
from wildfire.features.tabular.gap_days import touches_flagged
from wildfire.guards import provenance
from wildfire.labels import weights
from wildfire.labels.detections import epoch
from wildfire.models import arms
from wildfire.pipelines import features as feature_pipeline
from wildfire.pipelines.fit import arm_rows, final_fit, forward_folds


def run(log=print) -> dict:
    provenance.require_clean()
    harness.check_split("out_of_fold")
    t0 = time.time()
    caps = config.load("models")["fit_row_caps"]
    table, columns = feature_pipeline.load_model_table()
    fitted, ref = arms.fitted_arms(), arms.reference_arm()
    clim_col = columns[ref][0]
    e0 = epoch()
    parts, pis = [], {}
    for year, years in forward_folds():
        fit_mask = table.year.isin(years).values
        month = pd.to_datetime(e0 + pd.to_timedelta(table.day.values, "D")).month
        common = (table.year == year).values & (month <= 7)
        for a in fitted:
            common &= arm_rows(table, columns[a])
        H = table[common].reset_index(drop=True)
        d = dict(cell=H.cell.values, day=H.day.values, year=H.year.values, week=H.week.values, state=H.state.values,
                 label=H.label.values, climatology=H[clim_col].values)
        ok_ref = fit_mask & ~np.isnan(table[clim_col].values)
        iso = calibration.Isotonic(year - 1).fit(table[clim_col].values[ok_ref], table.label.values[ok_ref], table.year.values[ok_ref])
        d["p_" + ref] = iso.transform(H[clim_col].values)
        pis[(year, ref)] = float(table.label.values[ok_ref].mean())
        for a in fitted:
            mod = final_fit(table, a, columns, fit_mask, cap=caps["out_of_fold"])
            d["p_" + a] = mod.predict(H[columns[a]].values)
            pis[(year, a)] = mod.fit_prevalence
            log("out-of-fold", year, a)
        parts.append(pd.DataFrame(d))
    O = pd.concat(parts, ignore_index=True)
    out = config.output_dir("results")
    O.to_pickle(out / "out_of_fold_predictions.pkl")
    y = O.label.values
    names = [ref] + fitted
    clusters = prior_shift.cluster_labels(O.year.values, O.week.values, None, config.load("evaluation")["clusters"]["scheme"])
    probs = {a: O["p_" + a].values for a in names}
    scores = {a: (O.climatology.values if a == ref else probs[a]) for a in names}

    corr = {a: np.zeros(len(O)) for a in names}
    w = np.ones(len(O))
    masked = np.unique(table.cell.values[table.t2m.isna().values])
    end = pd.Timestamp(config.load("features")["tabular"]["window_end"])
    flagged = weights.flagged_days()
    lookback = weights.gap_lookback()
    season = weights.season_months()
    cf = weights.cell_frame()
    for year, _ in forward_folds():
        m = O.year.values == year
        tr = table[table.year < year]
        fr = weights.frame_sizes(sorted(tr.year.unique()), gap_days=flagged, lookback=lookback)
        rate = {st: float(tr[(tr.state == st) & (tr.label == 1)].shape[0] / sum(v for k, v in fr.items() if k.startswith(st + "|")))
                for st in tr.state.unique()}
        pit = np.array([rate[s] for s in O.state.values[m]])
        for a in names:
            corr[a][m] = prior_shift.prior_shift_probs(probs[a][m], pit, pis[(year, a)]) if (year, a) in pis else np.nan
        frame = {}
        for st, g in cf.groupby("state"):
            if st not in season:
                continue
            nc = int((~g.cell.isin(set(masked))).sum())
            days = pd.date_range(f"{year}-01-01", f"{year}-07-31")
            dn = np.array([(d_ - e0).days for d_ in days])
            ok = np.isin(days.month, season[st]) & ~touches_flagged(dn, flagged, lookback)
            frame[f"{st}|{year}"] = nc * int(ok.sum())
        w[m] = prior_shift.design_weights(y[m], weights.strata(O[m]), frame)
    before = harness.score_arms(O, scores, probs, ref, clusters)
    after = harness.score_arms(O, scores, corr, ref, clusters, weights=w)
    tb = harness.result_table(before, names, ref)
    ta = harness.result_table(after, names, ref)
    provenance.stamp_df(tb, out / "out_of_fold_before_prior_shift.csv")
    provenance.stamp_df(ta, out / "out_of_fold_after_prior_shift.csv")
    log("done", round(time.time() - t0))
    return dict(before=tb, after=ta, predictions=O)
