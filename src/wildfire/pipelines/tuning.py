"""Hyperparameter search: same budget for boosting and the network, rolling-origin validation, development years only."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import metrics, prior_shift
from wildfire.guards import provenance
from wildfire.models.fit import Model
from wildfire.models.network import Network, Standardiser
from wildfire.pipelines import dataset, ladder_fit

SPACES = {"boosting": "boosting_space", "network": "network_space"}
TUNED_FILE = "tuned.json"
LEVEL = "enriched"


def sample_trials(space: dict, n: int, seed: int) -> list[dict]:
    """n distinct random combinations of the search space."""
    combos = [dict(zip(space, v)) for v in itertools.product(*space.values())]
    order = np.random.default_rng(seed).permutation(len(combos))[:n]
    return [combos[i] for i in order]


def score_trial(p: np.ndarray, y: np.ndarray, fit_prevalence: float) -> dict:
    """Validation log loss after aligning the sampling rate to the validation rows, and AP."""
    aligned = prior_shift.prior_shift_probs(p, y.mean(), fit_prevalence)
    return dict(log_loss=metrics.log_loss(y, aligned), ap=metrics.average_precision(y, p))


def fold_matrices(ds: dataset.Dataset, year: int, cap: int) -> dict:
    """Capped fit rows and validation rows of one fold at the enriched level (the widest arm's inputs)."""
    fit_idx, val_idx = dataset.fit_rows(ds, year, cap), dataset.validation_rows(ds, year)
    cols = ladder_fit.columns_for(ds, "boosting")
    return dict(fit_idx=fit_idx, val_idx=val_idx, x=dataset.matrix(ds, fit_idx, cols, LEVEL, year),
                xv=dataset.matrix(ds, val_idx, cols, LEVEL, year))


def boosting_trial(ds, fm: dict, params: dict) -> dict:
    y = ds.table.label.values[fm["fit_idx"]]
    m = Model("boosting", params).fit(fm["x"], y)
    return score_trial(m.predict(fm["xv"]), ds.table.label.values[fm["val_idx"]], m.fit_prevalence)


def network_trial(ds, fm: dict, params: dict, year: int) -> dict:
    t = ds.table
    stop = t.year.values[fm["fit_idx"]] == year - 1
    w = ladder_fit.training_weights(t, fm["fit_idx"])
    y = t.label.values[fm["fit_idx"]]
    std = Standardiser(fm["x"][~stop])
    x, xs, xv = (std.apply(a.copy()) for a in (fm["x"][~stop], fm["x"][stop], fm["xv"]))
    m = Network(params, seed=config.load("baseline")["tuning"]["seed"]).fit(x, y[~stop], w[~stop], xs, y[stop], w[stop])
    return score_trial(m.predict(xv), t.label.values[fm["val_idx"]], m.fit_prevalence)


def choose(records: pd.DataFrame, kind: str) -> dict:
    """Trial with the lowest mean validation log loss (AP breaks ties)."""
    mean = records[records.kind == kind].groupby("trial")[["log_loss", "ap"]].mean().reset_index()
    best = mean.sort_values(["log_loss", "ap"], ascending=[True, False]).iloc[0]
    return json.loads(records[(records.kind == kind) & (records.trial == best.trial)].params.iloc[0])


def run(log=print) -> dict:
    """Run every trial on every tuning fold, record all trials, and write the chosen settings."""
    provenance.require_clean()
    cfg = config.load("baseline")
    ds = dataset.load_dataset()
    trials = {k: sample_trials(cfg["tuning"][space], cfg["tuning"]["trials"], cfg["tuning"]["seed"]) for k, space in SPACES.items()}
    records = []
    for year in cfg["validation"]["tuning_years"]:
        fm = fold_matrices(ds, year, cfg["tuning"]["row_cap"])
        for kind, plist in trials.items():
            for i, params in enumerate(plist):
                r = boosting_trial(ds, fm, params) if kind == "boosting" else network_trial(ds, fm, params, year)
                records.append(dict(kind=kind, trial=i, params=json.dumps(params), year=year, **r))
                log(kind, "trial", i, year, round(r["log_loss"], 5), round(r["ap"], 4))
    out = config.output_dir("results")
    table = pd.DataFrame(records)
    provenance.stamp_df(table, out / "tuning_trials.csv")
    tuned = {k: choose(table, k) for k in SPACES}
    (out / TUNED_FILE).write_text(json.dumps(tuned, indent=1))
    return tuned


def load_tuned() -> dict:
    return json.loads((config.output_dir("results") / TUNED_FILE).read_text())
