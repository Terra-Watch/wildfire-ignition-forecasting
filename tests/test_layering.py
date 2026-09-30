from pathlib import Path


def test_library_does_not_import_research_repo():
    src = Path(__file__).resolve().parents[1] / "src"
    for f in src.rglob("*.py"):
        assert "wildfire_research" not in f.read_text(), f
