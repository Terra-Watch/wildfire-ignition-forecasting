"""Risk map of one day. Library code: src/wildfire_forecast/risk_map.py. Arguments are passed through."""
import os, runpy, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("wildfire_forecast.risk_map", run_name="__main__", alter_sys=True)
