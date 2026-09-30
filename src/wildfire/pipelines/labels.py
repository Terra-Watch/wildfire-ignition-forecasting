"""Build labelled rows, cell-days and satellite gap-day flags from raw detections."""
from __future__ import annotations

import json
import time

from wildfire import config
from wildfire.features.tabular.gap_days import build_day_coverage
from wildfire.guards import validity
from wildfire.labels import detections, rows


def run(log=print) -> dict:
    t0 = time.time()
    out = config.output_dir("labels")
    det = detections.read_detections()
    log("detections after filters", len(det), round(time.time() - t0))
    coverage = build_day_coverage(det)
    coverage.to_csv(config.output_dir("features") / "day_coverage.csv", index=False)
    cells, meta = detections.load_grid()
    d = detections.assign_cells(det, cells, meta)
    cd = detections.cell_day_table(d, cells)
    cd.to_csv(out / "cell_days.csv", index=False)
    labelled, season = rows.build_rows(cd, cells)
    labelled.to_csv(out / "rows.csv", index=False)
    (out / "season.json").write_text(json.dumps(season, sort_keys=True))
    violations = validity.negatives_clear_of_detections(labelled, cd)
    assert violations == 0, f"{violations} negatives fall within the exclusion window after a detection"
    log("rows", len(labelled), "season", season, "runtime_s", round(time.time() - t0))
    return dict(rows=len(labelled), negatives_in_exclusion_window=violations)
