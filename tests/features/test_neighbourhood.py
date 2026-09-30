import numpy as np
import pandas as pd

from wildfire.features import neighbourhood as nb


def test_ring_mean_excludes_the_centre_and_uses_available_neighbours():
    r = np.full((5, 5), np.nan)
    r[2, 2] = 100.0                      # centre: must not count
    r[1, 2], r[3, 2] = 1.0, 3.0          # two available neighbours
    assert nb.ring_stat(r, 1)[2, 2] == 2.0
    assert np.isnan(nb.ring_stat(np.full((3, 3), np.nan), 1)[1, 1])


def test_edge_cells_use_the_neighbours_that_exist():
    r = np.ones((4, 4))
    m = nb.ring_stat(r, 1)
    assert m[0, 0] == 1.0 and m[3, 3] == 1.0                 # corners still have a mean
    assert nb.ring_stat(r, 1, total=True)[0, 0] == 3.0       # 3 neighbours at a corner
    assert nb.ring_stat(r, 1, total=True)[1, 1] == 8.0       # 8 in the interior
    assert nb.ring_stat(np.ones((5, 5)), 2, total=True)[2, 2] == 24.0    # the 24-cell ring


def test_raster_leaves_positions_without_a_cell_empty():
    cells = pd.DataFrame(dict(ix=[0, 2], iy=[1, 1]))
    r = nb.to_raster(np.array([5.0, 7.0]), cells, (3, 3))
    assert r[0, 1] == 5.0 and r[2, 1] == 7.0 and np.isnan(r[1, 1])


def test_recent_onset_counts_never_include_the_target_day():
    onsets = pd.DataFrame(dict(cell=[0, 0, 0], day=[100, 120, 130]))
    c = nb.recent_onset_counts(onsets, 1, np.array([100, 101, 129, 130, 131, 160]))[:, 0]
    assert c.tolist() == [0, 1, 1, 1, 2, 0]      # window = the 28 days before each target day; 130 is not seen on day 130


def test_column_names_follow_the_settings():
    names = nb.column_names()
    assert len(names) == 12 and names[0] == "ring_8_tree_cover" and names[-1] == "ring_24_onsets_recent"
