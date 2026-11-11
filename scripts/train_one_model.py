"""Train and score one model (one run folder); called by the queues. Library code: src/wildfire_forecast/harness.py. Arguments are passed through."""
import os, runpy, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
runpy.run_module("wildfire_forecast.harness", run_name="__main__", alter_sys=True)
