"""Fit every arm of the ladder for one validation year and predict its validation rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import calibration, prior_shift
from wildfire.labels import weights
from wildfire.models import arms, diagnostics
from wildfire.models.fit import Model
from wildfire.models.network import Network
from wildfire.pipelines import dataset

TWO_LEVEL = "two_levels"


def arm_levels() -> list[tuple[str, str]]:
    """(arm, level) for every ladder arm: two levels for the arms that have them, one otherwise."""
    cfg = config.load("baseline")["arms"]
    out = []
    for arm in cfg["ladder"]:
        out += [(arm, lv) for lv in dataset.LEVELS] if arm in cfg[TWO_LEVEL] else [(arm, "tabular")]
    return out


def arm_name(arm: str, level: str) -> str:
    two = arm in config.load("baseline")["arms"][TWO_LEVEL]
    return f"{arm}@{level}" if two else arm


def columns_for(ds: dataset.Dataset, arm: str) -> list[str]:
    """Tabular input columns of an arm; tabular_network and boosting read the widest arm's inputs."""
    if arm == "tabular_network":
        arm = "logistic_with_burn"
    return dataset.input_columns(ds, arm)


def training_weights(table: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """Design weights of the fit rows (per state-year stratum, from the eligible cell-days)."""
    rows = table.iloc[idx]
    frame = weights.frame_sizes(sorted(rows.year.unique()), gap_days=weights.flagged_days(), lookback=weights.gap_lookback())
    return prior_shift.design_weights(rows.label.values, weights.strata(rows), frame)


def fit_network(ds, fit_idx, val_idx, cols, level, year, params, log) -> tuple[dict, dict]:
    """One network per seed: fit on the years before the last fit year, stop early on the last fit year."""
    t = ds.table
    stop = t.year.values[fit_idx] == year - 1
    w = training_weights(t, fit_idx)
    x = dataset.matrix(ds, fit_idx, cols, level, year)
    xv = dataset.matrix(ds, val_idx, cols, level, year)
    y = t.label.values[fit_idx]
    preds, prevalence = {}, {}
    for seed in range(config.load("models")["network"]["seeds"]):
        m = Network(params, seed).fit(x[~stop], y[~stop], w[~stop], x[stop], y[stop], w[stop])
        preds[f"#{seed}"], prevalence[f"#{seed}"] = m.predict(xv), m.fit_prevalence
        log("network", level, year, "seed", seed, round(m.best_validation_loss, 5))
    return preds, prevalence


def fit_standard(ds, fit_idx, val_idx, cols, level, year, kind: str, params: dict) -> tuple[np.ndarray, float]:
    """Logistic or boosting on the fit rows; returns validation probabilities and the fit prevalence."""
    x = dataset.matrix(ds, fit_idx, cols, level, year)
    m = Model(kind, params).fit(x, ds.table.label.values[fit_idx])
    return m.predict(dataset.matrix(ds, val_idx, cols, level, year)), m.fit_prevalence


def climatology_arm(ds, year, val_idx) -> tuple[np.ndarray, np.ndarray, float]:
    """Isotonic-calibrated same-cell same-week rate from the years before `year`."""
    t = ds.table
    col = arms.reference_arm()
    clim = ds.tabular_columns[col][0]
    fit = (t.year.values < year) & t[clim].notna().values
    iso = calibration.Isotonic(year - 1).fit(t[clim].values[fit], t.label.values[fit], t.year.values[fit])
    score = t[clim].values[val_idx]
    return score, iso.transform(score), float(t.label.values[fit].mean())


def diagnostic_arms(ds, fit_idx, val_idx, boosting_params) -> dict:
    """coords_only, cell_id_only and static_only: what a model learns without any daily information."""
    t = ds.table
    y = t.label.values[fit_idx]
    out = {}
    xc = diagnostics.coordinate_matrix(ds.cells, t.cell.values[fit_idx])
    m = Model("boosting", boosting_params).fit(xc, y)
    out["coords_only"] = (m.predict(diagnostics.coordinate_matrix(ds.cells, t.cell.values[val_idx])), m.fit_prevalence)
    rate = diagnostics.cell_rates(t.cell.values[fit_idx], y, t.cell.values[val_idx])
    out["cell_id_only"] = (rate, float(y.mean()))
    static = dataset.feature_groups_static(ds)
    m = Model("boosting", boosting_params).fit(t[static].values[fit_idx].astype(np.float32), y)
    out["static_only"] = (m.predict(t[static].values[val_idx].astype(np.float32)), m.fit_prevalence)
    return out


def settings_for(arm: str, tuned: dict) -> tuple[str, dict]:
    """(model kind, hyperparameters) of a ladder arm: fixed for logistic arms, tuned for boosting."""
    spec = config.load("models")["arms"][arm]
    return spec["kind"], (tuned["boosting"] if spec["kind"] == "boosting" else spec["params"])


def validation_frame(ds, year: int, val_idx: np.ndarray) -> pd.DataFrame:
    """Keys, true per-state rate and design weight of the validation rows."""
    t = ds.table
    rows = t.iloc[val_idx][["cell", "day", "year", "week", "state", "label"]].reset_index(drop=True)
    rate = weights.rates_before(t, year)
    rows["pi_true"] = [rate[s] for s in rows.state]
    masked = np.unique(t.cell.values[t.t2m.isna().values])
    rows["weight"] = prior_shift.design_weights(rows.label.values, weights.strata(rows), weights.january_to_july_frame(year, masked))
    return rows


def run_fold(ds: dataset.Dataset, year: int, tuned: dict, log=print) -> tuple[pd.DataFrame, dict]:
    """Predictions of every arm on the validation rows of `year`, and each arm's fit prevalence."""
    fit_idx, val_idx = dataset.fit_rows(ds, year), dataset.validation_rows(ds, year)
    out = validation_frame(ds, year, val_idx)
    ref = arms.reference_arm()
    prevalence = {}
    score, prob, prevalence[ref] = climatology_arm(ds, year, val_idx)
    out["score_" + ref], out["p_" + ref] = score, prob
    for arm, level in arm_levels():
        name = arm_name(arm, level)
        if arm == ref:
            continue
        if arm == "tabular_network":
            preds, prev = fit_network(ds, fit_idx, val_idx, columns_for(ds, arm), level, year, tuned["network"], log)
            for k, v in preds.items():
                out[f"p_{name}{k}"], prevalence[f"{name}{k}"] = v, prev[k]
            continue
        kind, params = settings_for(arm, tuned)
        out["p_" + name], prevalence[name] = fit_standard(ds, fit_idx, val_idx, columns_for(ds, arm), level, year, kind, params)
        log("fitted", name, year)
    for name, (p, prev) in diagnostic_arms(ds, fit_idx, val_idx, tuned["boosting"]).items():
        out["p_" + name], prevalence[name] = p, prev
    return out, prevalence
