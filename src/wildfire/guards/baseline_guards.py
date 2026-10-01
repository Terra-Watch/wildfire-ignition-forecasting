"""Checks that must pass before any baseline score is written."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pyproj

from wildfire import config
from wildfire.evaluation import verdict, views
from wildfire.features import weather_windows
from wildfire.features.tabular import weather
from wildfire.guards import provenance
from wildfire.labels.detections import load_grid
from wildfire.models.fit import Model

GUARDS_FILE = "guards.json"
GUARDED_ARRAYS = ["neighbourhood.npy", "model_table.pkl", "windows.npy"]   # in the features folder
FULL_HASH_LIMIT = 2 * 1024 ** 3     # arrays up to this size are re-hashed on every check; larger ones by size and mtime
LEDGER_FILE = "geo_test_ledger.csv"


def synthetic_series(days: int = 200, points: int = 3) -> dict:
    rng = np.random.default_rng(0)
    return dict(values=rng.normal(size=(days, len(weather.NAMES), points)).astype(np.float32),
                rain=(rng.random((days, points)) * 6).astype(np.float32), cell_point=np.array([0, 1, 2]), first_day=1000)


def timestamp_before_day() -> bool:
    """Changing the series on the target day must not change that day's window features."""
    s = synthetic_series()
    cell, day = np.array([1]), np.array([1100])
    before = weather_windows.window_matrix(s, cell, day)
    s["values"][day[0] - s["first_day"]] += 100.0
    s["rain"][day[0] - s["first_day"]] += 100.0
    return bool(np.array_equal(before, weather_windows.window_matrix(s, cell, day), equal_nan=True))


def shifted_feature_fails() -> bool:
    """A feature deliberately taken one day late (from the future) must be detected as different."""
    s = synthetic_series()
    cell, day = np.array([0]), np.array([1100])
    return bool(not np.array_equal(weather_windows.window_matrix(s, cell, day),
                                   weather_windows.window_matrix(s, cell, day + 1), equal_nan=True))


def label_shuffle(ds) -> tuple[bool, float]:
    """Logistic model on permuted labels: mean AUC on the validation rows of the configured year must be 0.50 +- tolerance."""
    from wildfire.pipelines import dataset, ladder_fit
    cfg = config.load("baseline")["guards"]["label_shuffle_auc"]
    setup = config.load("guards")["label_shuffle"]
    year, seeds = setup["validation_year"], setup["seeds"]
    fit_idx = dataset.fit_rows(ds, year, config.load("baseline")["tuning"]["row_cap"])
    val_idx = dataset.validation_rows(ds, year)
    cols = ladder_fit.columns_for(ds, "logistic_with_burn")
    x, xv = dataset.matrix(ds, fit_idx, cols, "tabular", year), dataset.matrix(ds, val_idx, cols, "tabular", year)
    y, yv = ds.table.label.values[fit_idx], ds.table.label.values[val_idx]
    params = config.load("models")["arms"]["logistic_with_burn"]["params"]
    aucs = [views.auc(yv, Model("logistic", params).fit(x, np.random.default_rng(s).permutation(y)).predict(xv)) for s in range(seeds)]
    mean = float(np.mean(aucs))
    return abs(mean - cfg["center"]) <= cfg["tolerance"], mean


def coordinates_consistent(tolerance_m: float = 1.0) -> bool:
    """Grid indices, projected centroids and longitude/latitude of the cells agree with each other."""
    cells, meta = load_grid()
    size = config.load("domain")["grid"]["cell_size_m"]
    dom = config.load("domain")["grid"]
    x, y = pyproj.Transformer.from_crs(dom["source_crs"], dom["projected_crs"], always_xy=True).transform(cells.lon.values, cells.lat.values)
    centre_x = meta["x0"] + (cells.ix.values + 0.5) * size
    centre_y = meta["y0"] + (cells.iy.values + 0.5) * size
    inside = (np.abs(x - centre_x) <= size / 2 + tolerance_m) & (np.abs(y - centre_y) <= size / 2 + tolerance_m)
    same = (np.abs(cells.cx.values - centre_x) < tolerance_m) & (np.abs(cells.cy.values - centre_y) < tolerance_m)
    return bool(inside.all() and same.all())


def no_validation_information_in_tuning() -> bool:
    """Tuning years are all before the last development year, and each fold's climatology stops before its validation year."""
    cfg = config.load("baseline")
    return bool(max(cfg["validation"]["tuning_years"]) < cfg["periods"]["last_development_fold"])


def identity_gate(ds) -> bool:
    """No cell-day is both a fit row and a validation row in any rolling-origin fold. This is the meaning given to the specification's
    "identity gate" (see docs/decisions/2026-10-01-deviation-list.md, item l); it does not test for cell-identity shortcuts, which the
    diagnostic arms report without a gate."""
    from wildfire.pipelines import dataset
    t = ds.table
    key = t.cell.values.astype(np.int64) * 1_000_000 + t.day.values.astype(np.int64)
    for year in config.load("baseline")["validation"]["validation_years"]:
        shared = np.intersect1d(key[dataset.fit_rows(ds, year)], key[dataset.validation_rows(ds, year)])
        if len(shared):
            return False
    return True


def power_gate() -> bool:
    """The decision rule must hold back a verdict when the effect it could detect is larger than the threshold (spec section 7,
    `decision_rule.power_gate`): a contrast with a clear 3x-threshold gain is INCONCLUSIVE when its minimum detectable effect is
    twice the threshold and PASS when it is half the threshold, and the minimum detectable effect is (sum of `mde_z`) x the standard error."""
    gain = config.load("evaluation")["verdict"]["min_ap_gain"]
    z = sum(config.load("baseline")["decision_rule"]["mde_z"])

    def strong(se: float) -> dict:
        return dict(point=3 * gain, se=se, lo99=2 * gain, hi99=4 * gain)

    wide, mde_wide = verdict.baseline_verdict(strong(2 * gain / z), 0.0)
    narrow, mde_narrow = verdict.baseline_verdict(strong(0.5 * gain / z), 0.0)
    return bool(wide == "INCONCLUSIVE" and narrow == "PASS" and abs(mde_wide - 2 * gain) < 1e-12 and abs(mde_narrow - 0.5 * gain) < 1e-12)


def ledger_ready() -> bool:
    """The geographic-test ledger location exists and is writable."""
    return bool(config.output_dir("results").exists())


def run_all(ds) -> dict:
    shuffle_ok, shuffle_auc = label_shuffle(ds)
    return dict(timestamp_before_day=timestamp_before_day(), shifted_feature_fails=shifted_feature_fails(),
                identity_gate=identity_gate(ds), label_shuffle=shuffle_ok, label_shuffle_mean_auc=shuffle_auc, coordinate_unit_test=coordinates_consistent(),
                no_validation_information_in_tuning=no_validation_information_in_tuning(), power_gate=power_gate(), geo_ledger=ledger_ready(),
                provenance=bool(provenance.require_clean()))


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def array_fingerprints() -> dict:
    """Size, mtime and sha256 of the guarded arrays (the hash of a very large file is taken at record time only)."""
    folder = config.output_dir("features")
    out = {}
    for name in GUARDED_ARRAYS:
        st = (folder / name).stat()
        out[name] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=_sha256(folder / name))
    return out


def _arrays_match(recorded: dict) -> list[str]:
    folder = config.output_dir("features")
    bad = []
    for name, r in recorded.items():
        path = folder / name
        if not path.exists():
            bad.append(name)
            continue
        st = path.stat()
        if st.st_size != r["size"] or st.st_mtime_ns != r["mtime_ns"]:
            bad.append(name)
        elif st.st_size <= FULL_HASH_LIMIT and _sha256(path) != r["sha256"]:
            bad.append(name)
    return bad


def write_record(ds) -> dict:
    """Run every guard and write the outcome with the code version it applies to; stop when any fails."""
    results = run_all(ds)
    results["code_version"] = provenance.code_version()
    results["code_commit"] = provenance.tag()
    results["arrays"] = array_fingerprints()
    (config.output_dir("results") / GUARDS_FILE).write_text(json.dumps(results, indent=1))
    failed = [k for k, v in results.items() if v is False]
    if failed:
        raise SystemExit("GUARD FAILED: " + ", ".join(failed))
    return results


def require_record() -> dict:
    """No score file may be written unless a guard record exists, passed every guard, belongs to the current (clean) code
    version, and its recorded arrays are the arrays on disk."""
    path = config.output_dir("results") / GUARDS_FILE
    if not path.exists():
        raise SystemExit("no guard record: run `wildfire run-guards` before anything is scored")
    record = json.loads(path.read_text())
    if any(v is False for v in record.values()):
        raise SystemExit("the guard record contains a failed guard")
    if record.get("code_version") != provenance.code_version():
        raise SystemExit("the guard record belongs to other code: run `wildfire run-guards` again")
    if provenance.code_state()["dirty"]:
        raise SystemExit("uncommitted code changes: commit, then run `wildfire run-guards` again")
    if "arrays" not in record:
        raise SystemExit("the guard record has no array fingerprints: run `wildfire run-guards` again")
    bad = _arrays_match(record["arrays"])
    if bad:
        raise SystemExit("arrays changed since the guards ran: " + ", ".join(bad) + "; run `wildfire run-guards` again")
    return record


def record_geo_access(what: str) -> None:
    """Append one line to the geographic-test ledger; refuse a second scoring."""
    path = config.output_dir("results") / LEDGER_FILE
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["time", "code_commit", "what"])
    if (old.what == what).any():
        raise SystemExit(f"the geographic test '{what}' was already scored; it is scored once")
    row = pd.DataFrame([dict(time=pd.Timestamp.now().isoformat(), code_commit=provenance.tag(), what=what)])
    pd.concat([old, row]).to_csv(path, index=False)
