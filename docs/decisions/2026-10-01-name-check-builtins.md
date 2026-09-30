# Name check allows the numeric function and ERA5 wind names
Decision: the built-in numeric function call and the wind variable names `u10` and `v10` are not treated as labels. Real stage, attempt and version labels are still flagged; a test covers both.
Why: the check should catch labels, not ordinary code.
