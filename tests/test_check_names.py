import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_names", Path(__file__).resolve().parents[1] / "scripts" / "check_names.py")
check_names = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_names)

FORBIDDEN = ["pi" + "lot", "pha" + "se_2", "au" + "dit", "rou" + "nd_3", "v" + "2", "experi" + "ments", "R" + "1"]
# A base64 image payload that contains label-like tokens by chance (built from fragments so this file does not contain them).
PAYLOAD = "iVBORw0KGgo+" + "P" + "3/" + "V" + "0AAAANSUhEUgAAAxg="
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


def _notebook(tmp_path, source="x = 1\n", text="ok\n", png=PAYLOAD):
    import json
    nb = {"cells": [{"cell_type": "code", "source": [source], "metadata": {}, "outputs": [
        {"output_type": "stream", "name": "stdout", "text": [text]},
        {"output_type": "display_data", "data": {"text/plain": ["<Figure>"], "image/png": png}, "metadata": {}}]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb))
    return path


def test_a_base64_image_output_passes_but_label_like_text_in_a_notebook_still_fails(tmp_path):
    assert check_names.bad_tokens(PAYLOAD, is_code=True)     # the payload alone would be flagged
    assert not check_names.check_file(_notebook(tmp_path))
    assert check_names.check_file(_notebook(tmp_path, source=f"# {FORBIDDEN[0]} run\n"))               # source cell
    assert check_names.check_file(_notebook(tmp_path, text=f"{FORBIDDEN[2]} done\n"))                   # stream output
    p = _notebook(tmp_path)
    import json
    nb = json.loads(p.read_text())
    nb["cells"][0]["outputs"][1]["data"]["text/plain"] = [f"{FORBIDDEN[0]}"]                          # text output next to an image
    nb["cells"][0]["outputs"][1]["data"]["text/html"] = [f"<p>{FORBIDDEN[2]}</p>"]
    nb["cells"][0]["metadata"] = {"note": FORBIDDEN[0]}
    p.write_text(json.dumps(nb))
    problems = check_names.check_file(p)
    assert any("text/plain" in x for x in problems) and any("text/html" in x for x in problems) and any("metadata" in x for x in problems)


def test_only_the_image_mime_types_are_skipped(tmp_path):
    import json
    p = _notebook(tmp_path)
    nb = json.loads(p.read_text())
    nb["cells"][0]["outputs"][1]["data"]["application/json"] = PAYLOAD
    p.write_text(json.dumps(nb))
    assert check_names.check_file(p)
