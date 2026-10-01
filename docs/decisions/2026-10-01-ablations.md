# Secondary, post hoc ablations
Decision: three ablations of enriched boosting are run and reported as secondary and post hoc. They were defined after the baseline tables existed (the tables were written 2026-10-01 19:05-19:13 IST; this record is later). None of them changes the baseline, the arm set, the declared contrasts, the verdicts or the freeze; none is used to choose a setting. They use the same folds, the same fit rows (`fit_rows` at the 2,000,000 cap), the same tuned boosting settings (`tuned.json`) and the same validation rows as `boosting@enriched`.

| Name | Inputs | Question |
|---|---|---|
| A, `A_no_soil_anomaly` | the 42 tabular inputs without `sw1_anom` and `sw2_anom`, plus all enriched blocks | how much do the two soil-moisture anomalies, whose norm covers 2017-2023 (deviation c), matter |
| B1, `B_windows_only` | the 42 tabular inputs + weather windows (180) + window anomalies (17); no neighbourhood block | what the weather-window tier adds without the neighbourhood |
| B2, `B_neighbourhood_only` | the 42 tabular inputs + neighbourhood block (12); no weather windows or anomalies | what the neighbourhood tier adds without the windows |

- Each is compared with `boosting@enriched` on the same validation rows with the same paired ISO-week cluster bootstrap and the same verdict code. The result tables are labelled secondary; a verdict there is descriptive and does not enter the baseline's decision rule.
- Burn-free arms are not run (the sealed season is unavailable).
- A is a sensitivity check for deviation (c); a small difference supports, but does not prove, that the look-ahead in the 2017-2023 norm is not driving the result. It does not remove the deviation.
- Code: `pipelines/ablations.py`, settings in `configs/ablations.yaml`; outputs `ablation_*.parquet`, `ablation_metrics.csv`, `ablation_contrasts.csv` under the results folder. Run after the guard record is valid for the committed code, one fit at a time.
Why: the owner asked for a sensitivity analysis of the soil-anomaly inputs and, optionally, a split of the enriched blocks, after the baseline numbers were known. Recording them as post hoc keeps them from being read as part of the pre-declared design.

**Reporting note.** The verdict labels that the contrast code attaches to these rows (PASS, FAIL, INCONCLUSIVE) are descriptive and meaningless for removal contrasts, because the decision rule is built for a gain over a reference, not for the effect of taking inputs away. The paper reports ΔAP and its 99% interval only, and no verdict, for the ablations.
