# Network early stopping uses the last fit year
Decision: for each fold the network trains on the fit years before the last one and stops early on the last fit year; the best epoch is kept.
Why: a time-ordered hold-out inside the fit years, with no information from the validation year. The network therefore trains on one year fewer than the other arms.
