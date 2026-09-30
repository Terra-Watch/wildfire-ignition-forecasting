# Limitations

## Purpose
State plainly what the baseline can and cannot claim.

## Inputs
The results in `baseline_results.md` and the decision records in `docs/decisions/`.

## What the results do and do not support
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
| Weather resolution | About 9 km, so several 5 km cells share one weather cell. |
| Imagery | Not built; no imagery is in the data. |
| Statistical power | About 150 weekly clusters over five validation years; a contrast whose minimum detectable effect exceeds 0.010 is INCONCLUSIVE by rule. |

## Outputs
None; this page is a reading guide.

## How to re-run
Not applicable. Update it whenever a result changes what can be claimed.

## Known limits
This list is itself limited to what was checked; unknown weaknesses are not listed.
