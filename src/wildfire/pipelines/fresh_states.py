"""Fresh geographic test: states outside the fit domain, scored once, with a ledger of every access."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.features.tabular.build import build_features
from wildfire.guards import baseline_guards, provenance
from wildfire.labels import detections, rows
from wildfire.pipelines import dataset, enriched, features, ladder_fit, tuning

GROUP = "fresh"
TABLE = "fresh_table"
PREFIX = "fresh_"


def fresh_states() -> list[str]:
    """States of the fresh test that have cells available (see the scope decision and the completeness addendum)."""
    cfg = config.load("baseline")["periods"]["fresh_geographic_test"]
    return cfg["states_always"] + cfg["states_if_fire_data_complete"]


def build_rows(log=print) -> pd.DataFrame:
    """Labelled rows of the fresh states, drawn with their own reference season and the same negative rule."""
    out = config.output_dir("labels")
    cells, _ = detections.load_grid()
    cd = pd.read_csv(out / "cell_days.csv")
    labelled, season = rows.build_rows(cd, cells, roles=config.load("domain")["fresh_test_roles"], bands=("A",))
    labelled = labelled[labelled.state.isin(fresh_states())]
    labelled.to_csv(out / "fresh_rows.csv", index=False)
    log("fresh rows", labelled.groupby(["state", "kind"]).size().to_dict())
    return labelled


def build_dataset(log=print) -> None:
    """Features, model table and enriched arrays of the fresh rows (same code as the development rows)."""
    feats, labels = config.output_dir("features"), config.output_dir("labels")
    cd = pd.read_csv(labels / "cell_days.csv")
    table = build_features(labels / "fresh_rows.csv", feats / "day_coverage.csv", cd[cd.onset][["cell", "day"]], log=log)
    features.build_model_table(table, log, group=GROUP, name=TABLE)
    enriched.build_arrays(log, TABLE, PREFIX)


def validation_keys(dv: dataset.Dataset, idx: np.ndarray) -> pd.DataFrame:
    """Keys of the fresh rows; true rate is the configured population rate, weights are 1 (no frame for these states)."""
    keys = dv.table.iloc[idx][["cell", "day", "year", "week", "state", "label"]].reset_index(drop=True)
    keys["pi_true"] = config.load("baseline")["prior_shift"]["true_rate"]
    keys["weight"] = 1.0
    return keys


def score_once(log=print) -> None:
    """Fit on all development years, predict the fresh rows, write the predictions, record the ledger entries."""
    provenance.require_clean()
    ds = dataset.load_dataset()
    dv = dataset.load_dataset(TABLE, PREFIX)
    baseline_guards.require_passed(ds)
    for state in sorted(dv.table.state.unique()):
        baseline_guards.record_geo_access(state)
    last_year = config.load("baseline")["periods"]["last_development_fold"]
    fit_idx = dataset.fit_rows(ds, last_year + 1)
    val_idx = np.flatnonzero(dv.complete & dv.table[ds.tabular_columns["climatology_cell"][0]].notna().values)
    preds, prevalence = ladder_fit.predict_all(ds, fit_idx, dv, val_idx, last_year + 1, tuning.load_tuned(), log)
    out = pd.concat([validation_keys(dv, val_idx), preds], axis=1)
    folder = config.output_dir("results")
    out.to_parquet(folder / "fresh_predictions.parquet")
    pd.Series(prevalence).to_json(folder / "fresh_fit_prevalence.json")
    log("fresh predictions", out.shape)
