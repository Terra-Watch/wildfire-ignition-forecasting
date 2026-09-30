# Network trained with design weights
Decision: the network's loss weights each row by the design weight of its state-year stratum, scaled to mean 1; its fit prevalence is the weighted mean label, so the same prior-shift correction as for the other arms is applied.
Why: one calibration rule for every arm.
