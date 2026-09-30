# Evaluation

## Purpose
Say how arms are scored and how a difference is judged.

## Protocol
- Rolling origin: fit on the years before a validation year, score that year (2021 to 2025). No random splits.
- 2025 is a development fold: it was scored before, so it is not untouched.
- Sealed season (2026 January-July): unavailable (see the addendum).
- Fresh states (UK, AP): scored once, every access recorded in a ledger file.

## Metrics
Both ranking views everywhere: within-day spatial AUC (which cells are riskier on a day) and within-cell temporal AUC (which days are riskier for a cell). Also average precision (AP), log loss and Brier after correcting to the true per-state rate, top-k capture and lift, reliability by decile.

## Decision rule
PASS: AP gain of at least 0.010, 99% paired ISO-week cluster-bootstrap lower bound above 0, and no state worse than -0.03. FAIL: upper bound below 0.010. Otherwise INCONCLUSIVE. If the minimum detectable effect is above 0.010 the result is INCONCLUSIVE.

## Guards
Timestamp check, shifted-feature check, label-shuffle check, coordinate test, no validation information in tuning, clean tree with code hash, spec hash, geographic-test ledger. Results carry the code and spec commits.

## How to re-run
    wildfire baseline-tables
    wildfire score-fresh-states

## Known limits
- Absolute AP and lift depend on the sampled negatives; compare arms on the same rows.
- Five validation years give about 150 weekly clusters; small differences cannot be detected.
- The fresh-state test uses the configured population rate for probability correction, and unit weights.
