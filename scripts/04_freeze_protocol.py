"""Record the hash of every run and the selected settings before the blind states are scored. Library code: src/wildfire_forecast/freeze.py. Arguments are passed through."""
import os, runpy, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("wildfire_forecast.freeze", run_name="__main__", alter_sys=True)
