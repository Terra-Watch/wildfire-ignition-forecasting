"""Scores split by land-cover class. Library code: src/wildfire_forecast/landcover_split.py. Arguments are passed through."""
import os, runpy, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("wildfire_forecast.landcover_split", run_name="__main__", alter_sys=True)
