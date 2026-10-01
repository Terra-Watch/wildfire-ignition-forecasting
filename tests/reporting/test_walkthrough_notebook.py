import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "00_baseline_pipeline.ipynb"
ALLOWED = ("bw.", "w.", "display(", "import ", "from ", "raw", "w = ")


def test_notebook_code_cells_only_call_the_library():
    cells = [c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
    assert cells
    for c in cells:
        for line in "".join(c["source"]).splitlines():
            assert not any(k in line for k in ("np.", "pd.", "sklearn", "torch", "plt.")) or line.startswith("import"), line
