"""Fit every arm of the ladder for one validation year and predict its validation rows."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import calibration, prior_shift
from wildfire.labels import weights
from wildfire.models import arms, diagnostics
from wildfire.models.fit import Model
from wildfire.models.network import Network, Standardiser
from wildfire.pipelines import dataset

TWO_LEVEL = "two_levels"
DIAGNOSTIC_ARMS = ("coords_only", "cell_id_only", "static_only")


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


def validation_file(dv, val_idx, cols, level, year) -> np.ndarray:
    """The validation matrix of one input set, written once and memory-mapped (shared by every arm that reads it)."""
    name = f"validation_{year}_{level}_{len(val_idx)}_{hashlib.md5(','.join(cols).encode()).hexdigest()[:8]}.npy"
    path = config.output_dir("cache") / name
    return np.load(path, mmap_mode="r") if path.exists() else dataset.matrix_file(dv, val_idx, cols, level, year, path)


class FoldStore:
    """Finished models of one fold, one small file each, so that a stopped run resumes at model granularity.
    Each file records the code version that wrote it; a different version is refused, never mixed in."""

    def __init__(self, folder, code_version: str):
        self.folder, self.code_version = Path(folder), code_version
        self.folder.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.folder / (name.replace("@", "_at_").replace("#", "_seed_") + ".npz")

    def get(self, name: str) -> tuple[np.ndarray, float] | None:
        path = self._path(name)
        if not path.exists():
            return None
        with np.load(path) as f:
            if str(f["code_version"]) != self.code_version:
                raise SystemExit(f"{path.name} was written by code {str(f['code_version'])[:8]}, not {self.code_version[:8]}: "
                                 "delete the fold folder to refit it")
            return f["p"], float(f["prevalence"])

    def put(self, name: str, p: np.ndarray, prevalence: float) -> None:
        path = self._path(name)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, p=p, prevalence=prevalence, code_version=self.code_version)
        tmp.replace(path)

    def clear(self) -> None:
        for f in self.folder.iterdir():
            f.unlink()
        self.folder.rmdir()


def fit_network(ds, fit_idx, dv, val_idx, cols, level, year, params, log, store: FoldStore | None = None, name: str = "") -> tuple[dict, dict]:
    """One network per seed: fit on the years before the last fit year, stop early on the last fit year.
    With a store, finished seeds are read back and only the missing ones are fitted."""
    seeds = range(config.load("models")["network"]["seeds"])
    preds, prevalence = {}, {}
    for seed in seeds:
        done = store.get(f"{name}#{seed}") if store else None
        if done:
            preds[f"#{seed}"], prevalence[f"#{seed}"] = done
    if len(preds) == len(seeds):
        return preds, prevalence
    t = ds.table
    stop = t.year.values[fit_idx] == year - 1
    w = training_weights(t, fit_idx)
    y = t.label.values[fit_idx]
    x = dataset.matrix(ds, fit_idx[~stop], cols, level, year)
    std = Standardiser(x)
    std.apply(x)
    xs = std.apply(dataset.matrix(ds, fit_idx[stop], cols, level, year))
    xv = validation_file(dv, val_idx, cols, level, year)
    for seed in seeds:
        if f"#{seed}" in preds:
            continue
        m = Network(params, seed).fit(x, y[~stop], w[~stop], xs, y[stop], w[stop])
        preds[f"#{seed}"] = dataset.predict_chunks(m.predict, xv, transform=std.apply)
        prevalence[f"#{seed}"] = m.fit_prevalence
        if store:
            store.put(f"{name}#{seed}", preds[f"#{seed}"], prevalence[f"#{seed}"])
        log("network", level, year, "seed", seed, round(m.best_validation_loss, 5))
    return {k: preds[k] for k in sorted(preds, key=lambda s: int(s[1:]))}, prevalence


def fit_standard(ds, fit_idx, dv, val_idx, cols, level, year, kind: str, params: dict) -> tuple[np.ndarray, float]:
    """Logistic or boosting on the fit rows; returns validation probabilities and the fit prevalence."""
    y = ds.table.label.values[fit_idx]
    if kind == "logistic":                      # read from a file: the model standardises it chunk by chunk into float64
        path = config.output_dir("cache") / f"fit_{year}_{level}_{len(fit_idx)}_{hashlib.md5(','.join(cols).encode()).hexdigest()[:8]}.npy"
        x = dataset.matrix_file(ds, fit_idx, cols, level, year, path)
        m = Model(kind, params).fit(x, y)
        del x
        path.unlink()
    else:
        m = Model(kind, params).fit(dataset.matrix(ds, fit_idx, cols, level, year), y)
    return dataset.predict_chunks(m.predict, validation_file(dv, val_idx, cols, level, year)), m.fit_prevalence


def climatology_arm(ds, year, dv, val_idx) -> tuple[np.ndarray, np.ndarray, float]:
    """Isotonic-calibrated same-cell same-week rate from the years before `year`."""
    t = ds.table
    col = arms.reference_arm()
    clim = ds.tabular_columns[col][0]
    fit = (t.year.values < year) & t[clim].notna().values
    iso = calibration.Isotonic(year - 1).fit(t[clim].values[fit], t.label.values[fit], t.year.values[fit])
    score = dv.table[clim].values[val_idx]
    return score, iso.transform(score), float(t.label.values[fit].mean())


def diagnostic_arms(ds, fit_idx, dv, val_idx, boosting_params) -> dict:
    """coords_only, cell_id_only and static_only: what a model learns without any daily information."""
    t = ds.table
    y = t.label.values[fit_idx]
    out = {}
    xc = diagnostics.coordinate_matrix(ds.cells, t.cell.values[fit_idx])
    m = Model("boosting", boosting_params).fit(xc, y)
    out["coords_only"] = (m.predict(diagnostics.coordinate_matrix(dv.cells, dv.table.cell.values[val_idx])), m.fit_prevalence)
    rate = diagnostics.cell_rates(t.cell.values[fit_idx], y, dv.table.cell.values[val_idx])
    out["cell_id_only"] = (rate, float(y.mean()))
    static = dataset.feature_groups_static(ds)
    m = Model("boosting", boosting_params).fit(dataset.take_columns(t, fit_idx, static), y)
    out["static_only"] = (m.predict(dataset.take_columns(dv.table, val_idx, static)), m.fit_prevalence)
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


def predict_all(ds, fit_idx, dv, val_idx, year: int, tuned: dict, log=print, store: FoldStore | None = None) -> tuple[pd.DataFrame, dict]:
    """Fit every arm on `ds` rows `fit_idx` and predict rows `val_idx` of `dv`. `year` fixes the anomaly climatology
    (fit years are the years before it). Returns predictions (one column per arm) and each arm's fit prevalence.
    With a store, each model is saved as it finishes and models already in the store are not fitted again."""
    ref = arms.reference_arm()
    out = pd.DataFrame(index=range(len(val_idx)))
    score, prob, fit_prevalence = climatology_arm(ds, year, dv, val_idx)
    out["score_" + ref], out["p_" + ref] = score, prob
    prevalence = {ref: fit_prevalence}
    for arm, level in arm_levels():
        name = arm_name(arm, level)
        if arm == ref:
            continue
        if arm == "tabular_network":
            preds, prev = fit_network(ds, fit_idx, dv, val_idx, columns_for(ds, arm), level, year, tuned["network"], log, store, name)
            for k, v in preds.items():
                out[f"p_{name}{k}"], prevalence[f"{name}{k}"] = v, prev[k]
            continue
        done = store.get(name) if store else None
        if done is None:
            kind, params = settings_for(arm, tuned)
            done = fit_standard(ds, fit_idx, dv, val_idx, columns_for(ds, arm), level, year, kind, params)
            if store:
                store.put(name, *done)
            log("fitted", name, year)
        out["p_" + name], prevalence[name] = done
    diag = [store.get(n) if store else None for n in DIAGNOSTIC_ARMS]
    if any(d is None for d in diag):
        diag = diagnostic_arms(ds, fit_idx, dv, val_idx, tuned["boosting"])
        if store:
            for n, (p, prev) in diag.items():
                store.put(n, p, prev)
        diag = [diag[n] for n in DIAGNOSTIC_ARMS]
    for n, (p, prev) in zip(DIAGNOSTIC_ARMS, diag):
        out["p_" + n], prevalence[n] = p, prev
    return out, prevalence


def run_fold(ds: dataset.Dataset, year: int, tuned: dict, log=print, store: FoldStore | None = None) -> tuple[pd.DataFrame, dict]:
    """Predictions of every arm on the validation rows of `year`, with keys, true rates and design weights."""
    fit_idx, val_idx = dataset.fit_rows(ds, year), dataset.validation_rows(ds, year)
    keys = validation_frame(ds, year, val_idx)
    preds, prevalence = predict_all(ds, fit_idx, ds, val_idx, year, tuned, log, store)
    return pd.concat([keys, preds], axis=1), prevalence
