# Spec stamp
Decision: the stamped spec hash is the last commit that touched configs/baseline.yaml or docs/baseline_spec.md. The first scoring run writes it to the results folder; later runs refuse to start if either file differs from that commit.
Why: makes post-hoc edits to the pre-declared plan visible.
