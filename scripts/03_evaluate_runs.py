"""Evaluate every run: metrics per split and stratum, seed summaries, paired tests. Library code: src/wildfire_forecast/evaluate.py. Arguments are passed through."""
import os, runpy, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("wildfire_forecast.evaluate", run_name="__main__", alter_sys=True)
