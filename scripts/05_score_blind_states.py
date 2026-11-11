"""Score the blind states once, after the freeze (needs WILDFIRE_FINAL=1). Library code: src/wildfire_forecast/score_blind.py. Arguments are passed through."""
import os, runpy, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("wildfire_forecast.score_blind", run_name="__main__", alter_sys=True)
