# Limitations

## Purpose
State plainly what the baseline can and cannot claim.

## Inputs
The results in `baseline_results.md` and the decision records in `docs/decisions/`.

## What the results do and do not support
**Provisional values (leakage review, 2026-10-05).** Every pooled five-fold value, every fold-2023 value and every probability of detection, false alarm ratio and critical success index in the docs and notebooks is provisional, for the three reasons in the last three "Provisional" rows below. No result was removed or hidden; corrected numbers will replace them when they exist.

| Topic | Limit |
|---|---|
| Skill inside the fit states | Measured on rolling-origin validation years 2021-2025 in the six fit states. It says how well days and cells are ranked there, not how well the model will do next season. |
| New states | Measured once, on UK and AP cells that were never used to fit. See `baseline_results.md` for how much of the skill carries over; the fit-state numbers must not be quoted for other regions. |
| Sealed season | Unavailable. The burned-area product has no items after 2025-07, so the 2026 January-July rows cannot be given burn inputs. 2025 is not used in its place, and 2025 is a development fold that was scored before, so it is not untouched. |
| Causes | There are no cause labels. The models say where and when detections start, not why. |
| Grain | 5 km cells and whole days. Nothing finer is claimed. |
| Labels | Detections from two satellites: fires that are small, brief or hidden by cloud are missed; some detections are not wildfires. |
| Absolute numbers | Negatives are a sample (20 per onset, uniform in the season). AP and lift depend on that sampling; compare arms on the same rows, do not read AP as a forecast score. Log loss and Brier are corrected to the true rate per state. |
| Place shortcut | A model can learn which cells burn often. The diagnostic arms (coordinates, cell rate, static layers) show how much skill needs only the place. |
| Tuning | Twelve random-search trials per model, selected on 2021-2024; 2025 never drove a choice. A larger search could change the settings. |
| Row caps | Each fit uses at most 2,000,000 rows (all onsets kept); tuning uses at most 600,000. |
| Anomaly inputs | Two of the 42 inputs use a soil-moisture climate from 2017-2023, which includes validation years 2021-2023. |
| Provisional: fold 2023 (soil-moisture norm) | Fold-2023 numbers use soil-moisture anomalies whose norm includes later 2023 weather. `sw1_anom` and `sw2_anom` are soil moisture minus a same-ISO-week mean over 2017-2023, so a row in 2023 carries a statistic that includes weather from later in 2023, after the day being predicted. Every fold-2023 value is provisional, and so is every pooled five-fold value, because the same norm also covers rows of 2021 and 2022 (fold 2024 and 2025 rows are fully past). Measured only for the extended boosting: +0.0010 AP, 99% CI -0.0003 to +0.0025 (`docs/decisions/2026-10-01-soil-anomaly-deviation.md`). |
| Provisional: tuning on scored rows | Hyperparameters of every model were chosen on the scored rows. The baseline's settings were chosen on validation years 2021-2024, which are also scored folds; the deep multimodal model's settings were chosen on the fold-2024 scored rows. No fold value is an untouched estimate, and fold 2023 uses settings chosen partly with later (2024) labels. |
| Provisional: alert threshold of fold 2023 | The alert threshold for fold 2023 was chosen on fold-2024 labels. Probability of detection, false alarm ratio and critical success index use one threshold per model and fold taken from the other fold (`thresholds()` in `deep_multimodal/src/deep_multimodal/evaluation.py`, kept as ported): fold 2024 uses the fold-2023 threshold (earlier labels), fold 2023 uses the fold-2024 threshold (later labels). Every probability of detection, false alarm ratio and critical success index is provisional. |
| Night-lights rank | The frozen night-lights input is a percentile rank over all grid cells, including the UK, AP, reserve and buffer cells. It uses no labels and is part of the reproduced inputs. |
| Neighbours | Fit rows use only fit cells as neighbours. Fresh-state rows may use neighbours from every cell, including fit-state cells next to the border. |
| Lost imagery | The raw imagery folder was deleted and is not in any archive; the baseline does not use imagery (see the decision record). |
| Weather resolution | About 9 km, so several 5 km cells share one weather cell. |
| Imagery | Not built; no imagery is in the data. |
| Statistical power | About 150 weekly clusters over five validation years; a contrast whose minimum detectable effect exceeds 0.010 is INCONCLUSIVE by rule. |

## Outputs
None; this page is a reading guide.

## How to re-run
Not applicable. Update it whenever a result changes what can be claimed.

## Known limits
This list is itself limited to what was checked; unknown weaknesses are not listed.
