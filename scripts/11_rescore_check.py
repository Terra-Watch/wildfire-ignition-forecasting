"""Re-score saved models and compare with the stored predictions. Library code: src/wildfire_forecast/rescore_check.py. Arguments are passed through."""
import os, runpy, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("wildfire_forecast.rescore_check", run_name="__main__", alter_sys=True)
