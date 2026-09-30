# No score without a passing guard record
Decision: every scoring entry point (tuning, rolling-origin, baseline tables, fresh-state scoring) first requires a guard record written by `wildfire run-guards`: all guards true and the record made for the current code version (last commit touching code or configuration). Without it the entry point stops.
Why: the plan says guards before any score; an earlier tuning start did not enforce this. That tuning run was discarded.
