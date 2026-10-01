"""Secondary, post hoc ablations of enriched boosting (docs/decisions/2026-10-01-ablations.md): same folds, rows and settings as
`boosting@enriched`, with some inputs removed. They do not change the baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import ladder
from wildfire.features import neighbourhood, weather_windows
from wildfire.features.tabular import weather
from wildfire.guards import baseline_guards, provenance
from wildfire.models.fit import Model
from wildfire.pipelines import baseline_tables, dataset, ladder_fit, rolling_origin, tuning


def settings() -> dict:
    return config.load("ablations")


def block_matrix(ds: dataset.Dataset, idx: np.ndarray, columns: list[str], blocks: list[str], year: int) -> np.ndarray:
    """Float32 input matrix: the tabular columns, then the chosen enriched blocks in their usual order."""
    parts = [dataset.take_columns(ds.table, idx, columns)]
    t = ds.table
    if "windows" in blocks:
        parts.append(ds.arrays["windows"][idx])
    if "anomalies" in blocks:
        anom = weather_windows.anomalies(ds.arrays["series"], dataset.anomaly_climatology(ds, year), t.cell.values[idx], t.day.values[idx])
        parts.append(anom.astype(np.float32))
    if "neighbourhood" in blocks:
        parts.append(ds.arrays["neighbourhood"][idx])
    return np.hstack(parts)


def block_width(columns: list[str], blocks: list[str]) -> int:
    sizes = dict(windows=len(weather_windows.window_names()), anomalies=len(weather.NAMES), neighbourhood=len(neighbourhood.column_names()))
    return len(columns) + sum(sizes[b] for b in blocks)


def block_file(ds, idx, columns, blocks, year, path) -> np.ndarray:
    """The validation matrix written in chunks and read back memory-mapped."""
    chunk = config.load("models")["chunks"]["matrix_file"]
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(idx), block_width(columns, blocks)))
    for start in range(0, len(idx), chunk):
        out[start:start + chunk] = block_matrix(ds, idx[start:start + chunk], columns, blocks, year)
    out.flush()
    return np.load(path, mmap_mode="r")


def ablation_columns(ds: dataset.Dataset, spec: dict) -> list[str]:
    return [c for c in ladder_fit.columns_for(ds, "boosting") if c not in spec["drop"]]


def fit_fold(ds, year: int, tuned: dict, spec: dict) -> tuple[np.ndarray, float]:
    """Boosting with the tuned settings on the fold's fit rows; returns validation probabilities and the fit prevalence."""
    fit, val = dataset.fit_rows(ds, year), dataset.validation_rows(ds, year)
    columns = ablation_columns(ds, spec)
    m = Model("boosting", tuned["boosting"]).fit(block_matrix(ds, fit, columns, spec["blocks"], year), ds.table.label.values[fit])
    path = config.output_dir("cache") / f"ablation_validation_{year}.npy"
    xv = block_file(ds, val, columns, spec["blocks"], year, path)
    p = dataset.predict_chunks(m.predict, xv)
    del xv
    path.unlink()
    return p, m.fit_prevalence


def prediction_path(name: str, year: int):
    return config.output_dir("results") / f"ablation_{name}_{year}.parquet"


def run(log=print) -> None:
    """Fit every ablation on every validation year; finished (ablation, year) pairs are kept and skipped on a rerun."""
    provenance.require_clean()
    baseline_guards.require_record()
    tuned = tuning.load_tuned()
    ds = dataset.load_dataset()
    dataset.clear_cache()
    for name, spec in settings()["ablations"].items():           # one ablation at a time, in the order of the configuration
        for year in config.load("baseline")["validation"]["validation_years"]:
            path = prediction_path(name, year)
            if path.exists():
                continue
            p, prevalence = fit_fold(ds, year, tuned, spec)
            base, _ = rolling_origin.load_year(year)
            val = dataset.validation_rows(ds, year)
            assert len(base) == len(val) and (base.cell.values == ds.table.cell.values[val]).all() and (base.day.values == ds.table.day.values[val]).all()
            pd.DataFrame({"p": p, "prevalence": prevalence}).to_parquet(path)
            log("saved", name, year, len(p))
            dataset.clear_cache()
    write_tables(log)


def pooled() -> tuple[pd.DataFrame, dict, dict]:
    """Baseline rows with the ablation columns added: rows, corrected probabilities and scores of the reference and ablations."""
    years = config.load("baseline")["validation"]["validation_years"]
    ref = settings()["reference"]
    frames, corrected, scores = [], {ref: [], }, {}
    names = list(settings()["ablations"])
    for n in names:
        corrected[n] = []
    for y in years:
        f, prev = rolling_origin.load_year(y)
        frames.append(f[["cell", "day", "year", "week", "state", "label", "pi_true", "weight", "p_" + ref]])
        corrected[ref].append(ladder.corrected_probabilities(f["p_" + ref].values, f.pi_true.values, prev[ref]))
        for n in names:
            a = pd.read_parquet(prediction_path(n, y))
            assert len(a) == len(f)
            corrected[n].append(ladder.corrected_probabilities(a.p.values, f.pi_true.values, float(a.prevalence.iloc[0])))
            frames[-1]["p_" + n] = a.p.values
    rows = pd.concat(frames, ignore_index=True)
    corrected = {k: np.concatenate(v) for k, v in corrected.items()}
    scores = {k: rows["p_" + k].values for k in corrected}
    return rows, corrected, scores


def write_tables(log=print) -> None:
    rows, corrected, scores = pooled()
    out = config.output_dir("results")
    ref = settings()["reference"]
    w = rows.weight.values
    table = pd.DataFrame([dict(arm=a, **ladder.arm_row(rows, scores[a], corrected[a], w)) for a in corrected])
    provenance.stamp_df(table, out / "ablation_metrics.csv")
    clusters = ladder.cluster_labels(rows)
    contrasts = pd.DataFrame([dict(contrast=f"{a}_vs_{ref}", **ladder.contrast(rows, scores, corrected, w, a, ref, clusters)) for a in settings()["ablations"]])
    provenance.stamp_df(contrasts, out / "ablation_contrasts.csv")
    log("ablation tables written", out)
