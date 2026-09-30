from wildfire.cli import main


def test_features_command(capsys):
    assert main(["features"]) == 0
    out = capsys.readouterr().out
    for name in ("tabular", "weather_windows", "neighbourhood", "imagery_summary"):
        assert name in out
