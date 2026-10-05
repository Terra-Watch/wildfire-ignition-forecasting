import json

import numpy as np
import pandas as pd
import pytest

from wildfire.pipelines import in_season


def day_of(date: str) -> int:
    return (pd.Timestamp(date) - in_season.EPOCH).days


def test_mask_keeps_only_the_fire_season_months_of_the_rows_own_state():
    day = np.array([day_of("2022-02-10"), day_of("2022-08-10"), day_of("2022-11-20"), day_of("2022-02-10")])
    state = np.array(["OD", "OD", "OD", "MH"])
    season = {"OD": [1, 2, 3, 4, 5], "MH": [3, 4]}
    assert in_season.in_season_mask(day, state, season).tolist() == [True, False, False, False]


def make_source(tmp_path, negatives_outside=False):
    source = tmp_path / "source"
    (source / "features").mkdir(parents=True)
    (source / "labels").mkdir()
    days = [day_of(d) for d in ("2022-02-10", "2022-03-10", "2022-08-10", "2022-03-11")]
    table = pd.DataFrame({"cell": [1, 2, 3, 4], "day": days, "label": [0, 1, 1 if not negatives_outside else 0, 0], "state": ["OD"] * 4})
    table.to_pickle(source / "features" / "model_table.pkl")
    (source / "labels" / "season.json").write_text(json.dumps({"OD": [1, 2, 3]}))
    np.save(source / "features" / "windows.npy", np.arange(8, dtype=np.float32).reshape(2, 4))
    np.save(source / "features" / "neighbourhood.npy", np.arange(8, dtype=np.float32).reshape(4, 2))
    return source


def test_building_the_in_season_folder_drops_only_out_of_season_onsets_and_keeps_arrays_aligned(tmp_path):
    source = make_source(tmp_path)
    record = in_season.build_in_season_folder(source, tmp_path / "target")
    assert record["rows_before"] == 4 and record["rows_after"] == 3
    assert record["onsets_removed"] == 1 and record["negatives_removed"] == 0
    assert all(record["columns_identical_after_filter"].values())
    kept = np.load(tmp_path / "target" / "features" / "keep_index.npy")
    assert kept.tolist() == [0, 1, 3]
    assert np.load(tmp_path / "target" / "features" / "windows.npy").tolist() == [[0, 1, 3], [4, 5, 7]]
    assert np.load(tmp_path / "target" / "features" / "neighbourhood.npy").tolist() == [[0, 1], [2, 3], [6, 7]]


def test_building_stops_when_a_negative_row_lies_outside_the_season(tmp_path):
    source = make_source(tmp_path, negatives_outside=True)
    with pytest.raises(ValueError):
        in_season.build_in_season_folder(source, tmp_path / "target")
