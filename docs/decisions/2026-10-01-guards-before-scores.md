# No score without a passing guard record
Decision: every scoring entry point (tuning, rolling-origin, baseline tables, fresh-state scoring) first requires a guard record written by `wildfire run-guards`: all guards true and the record made for the current code version (last commit touching code or configuration). Without it the entry point stops.
Why: the plan says guards before any score; an earlier tuning start did not enforce this. That tuning run was discarded.

Addendum (array binding): the record also stores size, mtime and sha256 of the neighbourhood array, the model table and the windows array. A scoring entry point also stops when the code tree has uncommitted changes, or when an array on disk no longer matches the record (arrays up to 2 GB are re-hashed on every check; the larger windows array is checked by size and mtime). Why: the neighbourhood array was rebuilt after the first tuning start, so a record must name the array it covers.
