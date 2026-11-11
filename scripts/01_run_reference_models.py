"""Reference baselines (climatology, FWI, persistence) scored on every split. Library code: src/wildfire_forecast/run_reference_models.py. Arguments are passed through."""
import os, runpy, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("wildfire_forecast.run_reference_models", run_name="__main__", alter_sys=True)
