"""Metrics of the deep multimodal model and the extended boosting baseline on the same rows, from saved predictions only (folds 2024 and 2023)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deep_multimodal import bootstrap, metrics

FOLDS = (2024, 2023)
MODELS = {"baseline": "Frozen baseline (extended boosting)", "deep": "Deep multimodal model (8-seed ensemble)"}
# Average precision recorded when the models were first scored; the saved predictions must reproduce these.
RECORDED_AVERAGE_PRECISION = {("baseline", 2024): 0.226737, ("baseline", 2023): 0.179750, ("deep", 2024): 0.2913381635, ("deep", 2023): 0.2531670707}
RECORDED_TOLERANCE = 1e-5          # predictions are stored in single precision


@dataclass
class FoldPredictions:
    year: int
    cell: np.ndarray
    day: np.ndarray
    week: np.ndarray
    state: np.ndarray               # state names per row
    label: np.ndarray
    weight: np.ndarray
    quiet: np.ndarray
    raw: dict                       # model -> raw probability (ranking score); also "climatology"
    probability: dict               # model -> probability corrected to the true fire rate


def load_fold_predictions(directory: Path, year: int) -> FoldPredictions:
    """Read predictions_<year>.npz and rebuild the per-row weight and true fire rate from the small per-state tables."""
    directory = Path(directory)
    z = np.load(directory / f"predictions_{year}.npz")
    rates = json.loads((directory / "fit_rates.json").read_text())[str(year)]
    state_index, label = z["state_index"], z["label"].astype(int)
    weight, true_rate = z["weight_table"][state_index, label], z["true_rate"][state_index]
    raw = {"deep": z["deep_probability"].astype(float), "baseline": z["baseline_probability"].astype(float), "climatology": z["climatology_probability"].astype(float)}
    fit_rate = {"deep": rates["deep_model"], "baseline": rates["baseline"], "climatology": rates["cell_climatology"]}
    corrected = {name: metrics.correct_prior_shift(raw[name], true_rate, fit_rate[name]) for name in raw}
    return FoldPredictions(year, z["cell"], z["day"], z["week"], z["state_names"][state_index], label, weight, z["quiet"], raw, corrected)


def thresholds(folds: dict) -> dict:
    """One alert threshold per model and scored fold: the corrected-probability threshold that maximises the weighted critical success index on the OTHER fold, never on the scored fold.

    Other-fold rule, kept as ported from the research code: fold 2024 is scored with the threshold found on fold 2023 (a year before it, so only earlier labels are used),
    and fold 2023 is scored with the threshold found on fold 2024 (a LATER year, so the fold-2023 threshold uses labels from after the scored year).
    The probability of detection, false alarm ratio and critical success index that depend on it are therefore labelled provisional (docs/MODEL.md, Limitations)."""
    out = {}
    for year, scored in folds.items():
        other = folds[[y for y in folds if y != year][0]]
        for model in MODELS:
            out[(model, year)] = metrics.best_threshold(other.label, other.weight, other.probability[model])
    return out


def evaluate(directory: Path, draws: int = 2000) -> dict:
    """Returns {"table": long table of metrics, "calibration": calibration curves, "checks": dictionary of consistency checks}."""
    folds = {year: load_fold_predictions(directory, year) for year in FOLDS}
    limits = thresholds(folds)
    rows, curves, checks = [], [], {"average_precision_reproduced": {}, "thresholds": {f"{m}_scored_on_{y}": v for (m, y), v in limits.items()}}
    for year, fold in folds.items():
        for model in MODELS:
            ap = metrics.average_precision(fold.label, fold.raw[model])
            recorded = RECORDED_AVERAGE_PRECISION[(model, year)]
            checks["average_precision_reproduced"][f"{model}_{year}"] = dict(recorded=recorded, recomputed=ap)
            if abs(ap - recorded) > RECORDED_TOLERANCE:
                raise ValueError(f"average precision {ap} of {model} on {year} differs from the recorded {recorded}")
        groups = [("all rows", np.ones(len(fold.label), bool)), ("quiet stratum", fold.quiet)] + [(f"state {s}", fold.state == s) for s in sorted(set(fold.state))]
        for name, mask in groups:
            for model in MODELS:
                values, curve = metrics.score_rows(fold.label[mask], fold.weight[mask], fold.raw[model][mask], fold.probability[model][mask], fold.probability["climatology"][mask],
                                                   fold.day[mask], fold.cell[mask], limits[(model, year)])
                rows += [dict(fold=year, rows_scored=name, model=model, metric=k, value=v, lower_95=np.nan, upper_95=np.nan) for k, v in values.items()]
                if name in ("all rows", "quiet stratum"):
                    curves.append(curve.assign(fold=year, rows_scored=name, model=model))
        for name, mask in groups[:2]:
            blocks = np.char.add(np.char.add(str(year), "-W"), np.char.zfill(fold.week[mask].astype(str), 2))
            result = bootstrap.paired_average_precision(fold.label[mask], blocks, {m: fold.raw[m][mask] for m in MODELS}, "baseline", draws)
            for model in MODELS:
                iv = result[model]["average_precision"]
                rows.append(dict(fold=year, rows_scored=name, model=model, metric="average_precision_interval", value=iv["point"], lower_95=iv["lower"], upper_95=iv["upper"]))
            iv = result["deep"]["difference_to_reference"]
            rows.append(dict(fold=year, rows_scored=name, model="deep minus baseline", metric="average_precision_difference", value=iv["point"], lower_95=iv["lower"], upper_95=iv["upper"]))
            checks[f"bootstrap_{year}_{name}"] = dict(blocks=result["blocks"], draws=draws)
    return {"table": pd.DataFrame(rows), "calibration": pd.concat(curves, ignore_index=True), "checks": checks}
