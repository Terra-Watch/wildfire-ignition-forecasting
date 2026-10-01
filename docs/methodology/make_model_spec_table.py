"""Build docs/MODEL_SPEC_TABLE.md from the code, the configuration files and `artifacts/results/tuned.json` (not from the specification text).
Parameter values of scikit-learn objects come from `get_params()` of the objects built exactly as `wildfire.models.fit.Model` builds them.
Run from the repository root with the .env variables set: python docs/methodology/make_model_spec_table.py"""
import json
import time

import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from wildfire import config
from wildfire.models import arms
from wildfire.pipelines import ladder_fit

R = config.output_dir("results")
tuned = json.loads((R / "tuned.json").read_text())
models, baseline, evaluation, features = (config.load(n) for n in ("models", "baseline", "evaluation", "features"))
lg, bs, net = models["logistic"], models["boosting"], models["network"]
cols = arms.arm_columns()
n_inputs = {a: len(c) for a, c in cols.items()}
widest = n_inputs["logistic_with_burn"]
enriched_extra = 180 + 17 + 12


def params(obj) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in obj.get_params().items())


lr_by_arm = {a: LogisticRegression(C=models["arms"][a]["params"]["C"], solver=lg["solver"], max_iter=lg["max_iter"], tol=lg["tol"]) for a in
             ("landcover_season", "fwi_season", "logistic_weather_static", "logistic_with_burn")}
hgb = HistGradientBoostingClassifier(random_state=bs["random_state"], early_stopping=bs["early_stopping"], **tuned["boosting"])
iso = IsotonicRegression(y_min=evaluation["calibration"]["clip"], y_max=1 - evaluation["calibration"]["clip"], increasing=True, out_of_bounds="clip")
nt = tuned["network"]
cap, cap_seed = baseline["validation"]["fit_row_cap"], baseline["validation"]["fit_row_cap_seed"]

out = f"""# Model specification table

Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `docs/methodology/make_model_spec_table.py` from the code, the configuration files and `artifacts/results/tuned.json`. Nothing here is taken from the text of `docs/baseline_spec.md`; where the two differ, the code and `tuned.json` are what produced the results (deviation i). Library versions in the environment: scikit-learn {sklearn.__version__}, PyTorch {torch.__version__}.

## Common to every fitted arm
- **Fit rows** (`dataset.fit_rows`): complete rows (every tabular input and every enriched feature present) of the years before the validation year, at most {cap:,} rows, all onsets kept, negatives thinned uniformly without replacement with `numpy.random.default_rng({cap_seed})` (`models.fit.thin`). The same rows are used by every arm and both levels of a fold. The fresh test fits on the years before 2026 with the same cap and seed.
- **Inputs:** the arm's tabular columns (`arms.arm_columns`, groups from `configs/features.yaml`), plus at the enriched level the 180 weather-window columns, 17 window anomalies and 12 neighbourhood columns ({enriched_extra} extra columns). `boosting` and `tabular_network` read the {widest} columns of `logistic_with_burn` (`ladder_fit.columns_for`).
- **Prior-shift correction after prediction** (`prior_shift.prior_shift_probs`): the fitted probability's odds are multiplied by [π_true / (1 − π_true)] / [π_fit / (1 − π_fit)], with π_true the per-state onset rate of the training years (the configured {baseline['prior_shift']['true_rate']} for the fresh states) and π_fit the fit prevalence of the arm; probabilities are clipped at 1e-15. Log loss and Brier use these corrected probabilities with the design weights; AP and the AUC views use the uncorrected scores.
- **Design weights** (`prior_shift.design_weights`, `weights.strata`): onsets weight 1; the negatives of a state-year stratum weigh (eligible cell-days of the stratum − onsets) / sampled negatives. They enter (a) the network loss, and (b) the evaluation of log loss and Brier. They do not enter the logistic or the boosting fit, which are unweighted.

## Arms
| Arm | Library and class | Inputs (columns) | Parameters actually used | Input scaling | Design weights in the fit | Optimiser and stopping | Random seeds |
|---|---|---|---|---|---|---|---|
| climatology_cell | `sklearn.isotonic.IsotonicRegression` on the column `climatology_cell` (same-cell, same-ISO-week onset rate of earlier years), wrapped in `evaluation.calibration.Isotonic` | 1 | {params(iso)}; fitted on fit-year rows with a non-missing climatology value, refuses rows after the calibration year limit (`max_train_year` = validation year − 1) | none | none | isotonic least squares (no iteration setting) | none |
| landcover_season | `sklearn.linear_model.LogisticRegression` (via `models.fit.Model`) | {n_inputs['landcover_season']} (land cover, season) | {params(lr_by_arm['landcover_season'])} | per column: mean and SD (float64, then float32) of the fit rows, SD plus {lg['scale_epsilon']}; applied in float32 in row chunks and handed to scikit-learn as float64 | none (unweighted fit) | lbfgs, stops at `max_iter` or `tol`; the 300-iteration cap is reached in the fits checked (deviation d) | none (deterministic) |
| fwi_season | the same class | {n_inputs['fwi_season']} (fire-weather codes, season) | {params(lr_by_arm['fwi_season'])} | the same | none | the same | none |
| logistic_weather_static | the same class | {n_inputs['logistic_weather_static']} (weather, codes, static, human, season) | {params(lr_by_arm['logistic_weather_static'])} | the same | none | the same | none |
| logistic_with_burn | the same class | {n_inputs['logistic_with_burn']} (the above plus two burn inputs) | {params(lr_by_arm['logistic_with_burn'])} | the same | none | the same | none |
| boosting | `sklearn.ensemble.HistGradientBoostingClassifier` | {widest} tabular, or {widest + enriched_extra} enriched | {params(hgb)}; values from `tuned.json`: {tuned['boosting']}; `random_state` and `early_stopping` from `configs/models.yaml` `boosting` | none (raw columns, float32) | none (unweighted fit) | gradient boosting for exactly `max_iter` = {tuned['boosting']['max_iter']} iterations (`early_stopping` False, so `n_iter_no_change`, `validation_fraction` and `scoring` are not used) | `random_state` = {bs['random_state']} (governs any random draw inside the scikit-learn fit, such as the row subsample used for the bin thresholds) |
| tabular_network | `torch.nn.Sequential` built by `models.network.build_net`, trained by `models.network.Network` | {widest} tabular, or {widest + enriched_extra} enriched | {nt['depth']} hidden layers: the first `Linear(n_inputs, {nt['width']})`, the next `Linear({nt['width']}, {nt['width']})`, each followed by `ReLU` and `Dropout({nt['dropout']})`; then `Linear({nt['width']}, 1)` (one logit, sigmoid at prediction); width {nt['width']}, depth {nt['depth']}, dropout {nt['dropout']}, learning rate {nt['learning_rate']}, weight decay {nt['weight_decay']} (`tuned.json`); batch size {net['batch_size']}, at most {net['max_epochs']} epochs (`configs/models.yaml` `network`) | per column mean and SD of the training rows (the fit rows without the last fit year), SD plus {lg['scale_epsilon']}, float32 (`Standardiser`) | design weights divided by their mean over the rows they belong to (training rows for the loss, last-fit-year rows for the stopping loss); batch loss = sum(weight × binary cross-entropy with logits) / sum(weight) (`weighted_log_loss`); fit prevalence = weighted mean label | `torch.optim.AdamW(lr={nt['learning_rate']}, weight_decay={nt['weight_decay']})`, PyTorch defaults for the other arguments (betas (0.9, 0.999), eps 1e-08); per epoch the rows are shuffled with `numpy.random.default_rng(seed)`; early stopping on the weighted log loss (same weighting) of the last fit year, which is held out of the training rows: the best epoch is kept (improvement of at least {net['min_improvement']}), training stops after {net['patience']} epochs without improvement | seeds 0 to {net['seeds'] - 1}: `torch.manual_seed(seed)` and `numpy.random.default_rng(seed)`; every seed is its own model and is reported separately (never averaged into one prediction); device `mps` when available else `cpu` (`pick_device`; MPS was available on this machine, the run logs do not record the device) |
| coords_only | `HistGradientBoostingClassifier`, same settings as `boosting` | 2 (projected cell centroid x, y) | as `boosting` | none | none | as `boosting` | `random_state` = {bs['random_state']} |
| cell_id_only | none (a lookup) | cell id | onset share of the cell among the fit rows; cells absent from the fit rows get the overall onset share | none | none | none | none |
| static_only | `HistGradientBoostingClassifier`, same settings as `boosting` | the 16 static columns (terrain, land cover, land structure) | as `boosting` | none | none | as `boosting` | `random_state` = {bs['random_state']} |

## Tuning (not part of the fitted arms)
Boosting and the network were tuned once at the enriched level on the widest arm's inputs, by random search with seed {baseline['tuning']['seed']}, {baseline['tuning']['trials']} trials each, on fits of at most {baseline['tuning']['row_cap']:,} rows, rolling-origin validation years 2021-2024, selection by the log loss aligned to the validation sample's onset share (AP breaks ties); the network was tuned with one seed (seed 0). Search spaces: `configs/baseline.yaml` (`boosting_space`, `network_space`). The chosen settings are in `artifacts/results/tuned.json`; the logistic C values are fixed in `configs/models.yaml`. Every trial is in `artifacts/results/tuning_trials.csv` (24 trials, each scored on 4 folds).

## Where each statement comes from
`src/wildfire/models/fit.py` (`Model`, `thin`), `src/wildfire/models/network.py` (`build_net`, `Standardiser`, `Network`, `weighted_log_loss`), `src/wildfire/models/diagnostics.py`, `src/wildfire/evaluation/calibration.py`, `src/wildfire/evaluation/prior_shift.py`, `src/wildfire/pipelines/ladder_fit.py` (`fit_standard`, `fit_network`, `settings_for`, `training_weights`, `climatology_arm`, `diagnostic_arms`), `src/wildfire/pipelines/dataset.py` (`fit_rows`), `configs/models.yaml`, `configs/baseline.yaml`, `configs/features.yaml`, `artifacts/results/tuned.json`.
"""
(config.REPO_ROOT / "docs" / "MODEL_SPEC_TABLE.md").write_text(out)
print(len(out), "characters written")
