# Validation rows and fit caps
Decision: every validation year uses January-July rows; every fit uses at most 2,000,000 training rows (tuning: 600,000), all onsets kept, same rows for every arm and level.
Why: the 2025 data stops at the end of July, so all years use the same months; the caps keep the enriched tables within memory, and equal rows keep the tabular-vs-enriched comparison fair.
