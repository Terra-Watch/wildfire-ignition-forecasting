"""Build the tabular feature table and the model table."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.features.tabular.build import build_features
from wildfire.labels.detections import day_number
from wildfire.models import arms

TABLE_META = ["cell", "day", "year", "week", "state", "label"]


def build_feature_table(log=print) -> pd.DataFrame:
    t0 = time.time()
    labels = config.output_dir("labels")
    feats = config.output_dir("features")
    cd = pd.read_csv(labels / "cell_days.csv")
    table = build_features(labels / "rows.csv", feats / "day_coverage.csv", cd[cd.onset][["cell", "day"]], log=log)
    table.to_pickle(feats / "tabular.pkl")
    log("features saved", table.shape, "runtime_s", round(time.time() - t0))
    return table


def build_model_table(features: pd.DataFrame | None = None, log=print) -> pd.DataFrame:
    """Main population (training group, onset/negative, primary band, through the window end) with the union of all arm
    inputs, as float32; also writes the per-arm column lists next to it."""
    feats = config.output_dir("features")
    R = pd.read_pickle(feats / "tabular.pkl") if features is None else features
    cols_by_arm = arms.arm_columns(R)
    cols = sorted({c for v in cols_by_arm.values() for c in v})
    end = day_number(config.load("features")["tabular"]["window_end"])
    m = (R.group == "train") & R.kind.isin(["onset", "negative"]) & (R.band == "A") & (R.day <= end)
    T = R.loc[m, TABLE_META + cols].copy()
    T["state"] = T.state.astype(str)
    for c in cols:
        T[c] = T[c].astype(np.float32)
    T["label"] = T.label.astype(np.int8)
    T = T.reset_index(drop=True)
    T.to_pickle(feats / "model_table.pkl")
    (feats / "arm_columns.json").write_text(json.dumps(cols_by_arm, indent=1))
    log("model table", T.shape, "years", T.year.value_counts().sort_index().to_dict())
    return T


def load_model_table() -> tuple[pd.DataFrame, dict]:
    feats = config.output_dir("features")
    return pd.read_pickle(feats / "model_table.pkl"), json.loads((feats / "arm_columns.json").read_text())
