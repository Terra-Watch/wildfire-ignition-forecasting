from wildfire.data import manifest


def test_describe_and_verify(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.csv").write_text("x\n")
    spec = {"paths": ["raw/*.csv"], "description": "d"}
    info = manifest.describe_dataset(tmp_path, spec)
    assert info["file_count"] == 1 and info["bytes"] == 2
    again = manifest.describe_dataset(tmp_path, spec)
    assert info["sha256"] == again["sha256"]
    (tmp_path / "raw" / "a.csv").write_text("y\n")
    assert manifest.describe_dataset(tmp_path, spec)["sha256"] != info["sha256"]


def test_artifact_manifest_describes_the_results_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "table.csv").write_text("a\n1\n")
    m = manifest.build(group="artifact_groups")
    assert m["root_env"] == "ARTIFACTS_ROOT" and m["datasets"]["results"]["file_count"] == 1
    assert m["datasets"]["labels"]["file_count"] == 0
