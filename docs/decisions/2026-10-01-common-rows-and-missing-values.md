# One row set for every arm
Decision: every arm is fitted and scored on the same rows: rows with no missing value in any tabular, window or neighbourhood input (and, for validation, a climatology value). Nothing is imputed.
Why: same rows, labels, folds and metrics for all arms. About 0.5% of rows are dropped, mostly where a weather cell is masked.
