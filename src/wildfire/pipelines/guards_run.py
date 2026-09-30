"""Run every guard on the current tables and write the guard record (read-only on the data)."""
from __future__ import annotations

from wildfire.guards import baseline_guards, provenance
from wildfire.pipelines import dataset


def run(log=print) -> dict:
    provenance.require_clean()
    ds = dataset.load_dataset()
    record = baseline_guards.write_record(ds)
    dataset.clear_cache()
    log("guards", record)
    return record
