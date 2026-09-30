import numpy as np
import pandas as pd

from wildfire.models import diagnostics


def test_coordinate_matrix_uses_the_cell_centroid():
    cells = pd.DataFrame(dict(cx=[10.0, 20.0, 30.0], cy=[1.0, 2.0, 3.0]))
    assert diagnostics.coordinate_matrix(cells, np.array([2, 0])).tolist() == [[30.0, 3.0], [10.0, 1.0]]


def test_cell_rates_come_from_fit_rows_only_and_unseen_cells_get_the_overall_share():
    r = diagnostics.cell_rates(np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 5]))
    assert r.tolist() == [0.5, 0.0, 0.25]
