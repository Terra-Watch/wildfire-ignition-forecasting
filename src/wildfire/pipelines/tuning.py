"""Hyperparameter search: same budget for boosting and the network, rolling-origin validation, development years only."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import metrics, prior_shift
from wildfire.guards import baseline_guards, provenance
from wildfire.models.fit import Model
from wildfire.models.network import Network, Standardiser
from wildfire.pipelines import dataset, ladder_fit

SPACES = {"boosting": "boosting_space", "network": "network_space"}
TUNED_FILE = "tuned.json"
TRIALS_FILE = "tuning_trials.jsonl"
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
                xv=ladder_fit.validation_file(ds, val_idx, cols, LEVEL, year))


def boosting_trial(ds, fm: dict, params: dict) -> dict:
    y = ds.table.label.values[fm["fit_idx"]]
    m = Model("boosting", params).fit(fm["x"], y)
    return score_trial(dataset.predict_chunks(m.predict, fm["xv"]), ds.table.label.values[fm["val_idx"]], m.fit_prevalence)


def network_trial(ds, fm: dict, params: dict, year: int) -> dict:
    t = ds.table
    stop = t.year.values[fm["fit_idx"]] == year - 1
    w = ladder_fit.training_weights(t, fm["fit_idx"])
    y = t.label.values[fm["fit_idx"]]
    std = Standardiser(fm["x"][~stop])
    x, xs = std.apply(fm["x"][~stop]), std.apply(fm["x"][stop])
    m = Network(params, seed=config.load("baseline")["tuning"]["seed"]).fit(x, y[~stop], w[~stop], xs, y[stop], w[stop])
    p = dataset.predict_chunks(m.predict, fm["xv"], transform=std.apply)
    return score_trial(p, t.label.values[fm["val_idx"]], m.fit_prevalence)


def choose(records: pd.DataFrame, kind: str) -> dict:
    """Trial with the lowest mean validation log loss (AP breaks ties)."""
    mean = records[records.kind == kind].groupby("trial")[["log_loss", "ap"]].mean().reset_index()
    best = mean.sort_values(["log_loss", "ap"], ascending=[True, False]).iloc[0]
    return json.loads(records[(records.kind == kind) & (records.trial == best.trial)].params.iloc[0])


def load_done(path) -> list[dict]:
    """Trial records already written (one JSON object per line)."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append_record(path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def run(log=print) -> dict:
    """Run every trial on every tuning fold, record all trials, and write the chosen settings.

    Every finished trial is appended to a file at once; a restart skips the trials already in it.
    """
    provenance.require_clean()
    baseline_guards.require_record()
    cfg = config.load("baseline")
    out = config.output_dir("results")
    path = out / TRIALS_FILE
    records = load_done(path)
    done = {(r["kind"], r["trial"], r["year"]): r for r in records}
    trials = {k: sample_trials(cfg["tuning"][space], cfg["tuning"]["trials"], cfg["tuning"]["seed"]) for k, space in SPACES.items()}
    ds = dataset.load_dataset()
    dataset.clear_cache()
    for year in cfg["validation"]["tuning_years"]:
        todo = [(k, i) for k, pl in trials.items() for i in range(len(pl)) if (k, i, year) not in done]
        if not todo:
            continue
        fm = fold_matrices(ds, year, cfg["tuning"]["row_cap"])
        for kind, i in todo:
            params = trials[kind][i]
            r = boosting_trial(ds, fm, params) if kind == "boosting" else network_trial(ds, fm, params, year)
            rec = dict(kind=kind, trial=i, params=json.dumps(params), year=year, **r)
            append_record(path, rec)
            records.append(rec)
            log(kind, "trial", i, year, round(r["log_loss"], 5), round(r["ap"], 4))
        dataset.clear_cache()
    table = pd.DataFrame(records)
    provenance.stamp_df(table, out / "tuning_trials.csv")
    tuned = {k: choose(table, k) for k in SPACES}
    (out / TUNED_FILE).write_text(json.dumps(tuned, indent=1))
    return tuned


def load_tuned() -> dict:
    return json.loads((config.output_dir("results") / TUNED_FILE).read_text())
