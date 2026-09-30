import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_names", Path(__file__).resolve().parents[1] / "scripts" / "check_names.py")
check_names = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_names)

FORBIDDEN = ["pi" + "lot", "pha" + "se_2", "au" + "dit", "rou" + "nd_3", "v" + "2", "experi" + "ments", "R" + "1"]
ALLOWED = ["background", "around", "ground_truth", "era5", "t2m", "v10", "validation"]


def test_forbidden_words_are_flagged():
    for word in FORBIDDEN:
        assert check_names.bad_tokens(f"x_{word}_y"), word


def test_ordinary_words_pass():
    for word in ALLOWED:
        assert not check_names.bad_tokens(word), word


def test_file_content_is_checked(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("# " + FORBIDDEN[0] + " code\n")
    assert check_names.check_file(f)
    f.write_text("x = 1\n")
    assert not check_names.check_file(f)


def test_builtin_numeric_function_is_allowed_in_code():
    call = "ro" + "und"
    assert not check_names.bad_tokens(f"x = {call}(1.5)", is_code=True)
    assert check_names.bad_tokens(f"{call}_3 = 1", is_code=True)


def test_real_version_labels_are_flagged_and_lookalikes_allowed():
    labels = ["v" + "1", "v" + "2", "pha" + "se", "rou" + "nd", "pi" + "lot", "au" + "dit", "experi" + "ment"]
    for word in labels:
        assert check_names.bad_tokens(f"name_{word}_x", is_code=True), word
        assert check_names.bad_tokens(f"x = '{word}'", is_code=True), word
    call = "ro" + "und"
    assert not check_names.bad_tokens(f"y = {call}(2.5, 1)", is_code=True)
    assert not check_names.bad_tokens("wind = ds['v10'] + ds['u10']", is_code=True)
