# Model

## What it is
A neural network that reads four kinds of input about one 5 km forest cell and one day and returns the **probability that a new fire starts in that cell on that day** (next-day fire occurrence). The output is a probability per cell and day, not an ignition point; a map of the probabilities over all cells is a hazard map. The final model is an ensemble of 8 networks (8 seeds) trained with hard labels only (fire start 0 or 1; no soft labels, no distillation from another model). Each network has 296,587 parameters.

## Inputs (details in DATA.md)
1. **Fire history**: detections in the 7 by 7 window of cells around the target cell over the previous 14 days (at most 32 occupied entries, each with 4 numbers).
2. **Weather sequence**: 60 days of 37 daily numbers (20 reanalysis channels and 17 anomalies).
3. **Day-ahead forecast tokens**: the 06 UTC forecast of the day before, valid on the target day: 5 variables at 12 hourly steps plus 20 daily statistics.
4. **Static features**: 21 time-invariant columns plus 2 burn-history columns, one token each.
5. **Calendar and own-cell values** (sine and cosine of the day of the year, three own-cell fire-history values, six engineered columns for a skip path).

## Architecture (width 64, 2 layers, 4 attention heads, dropout 0.1)
- **Fire-history transformer**: each occupied entry becomes a token (a linear map of its 4 numbers plus an embedding of its lag day and one of its position in the 7 by 7 window). One always-valid empty token is added so no row has every entry masked. Two self-attention layers refine the tokens; empty slots are masked.
- **Weather sequence stream**: a 2-layer recurrent network (GRU) reads the 60 days; its hidden state at the end of every fifth day gives 12 tokens (plus a learned segment embedding).
- **Forecast stream**: a 1D convolution over the 12 hourly steps, pooled to 6 tokens; the 20 daily statistics are added to every token. A day with no forecast is masked, never filled with zeros.
- **Static stream**: each static column value scales a learned vector, plus a learned shift and a column embedding; two self-attention layers mix the 23 tokens; their mean is the static summary.
- **Cross-attention fusion**: a query token is built from the static summary, the calendar values and three own-cell history values. Two cross-attention layers let the query attend over all stream tokens (each token carries a learned stream-type embedding). The query is rescaled and shifted by a learned function of the static summary and the calendar values; this conditioning starts at zero effect.
- **Output**: a small network on the fused query gives the logit; a gated skip path from six engineered columns (gate starts at zero) is added. The probability is the sigmoid of the logit.
- **Modality dropout** (training only): for 10% of rows each, the whole weather, forecast or static stream is hidden. The fire-history stream is never hidden.
- **Auxiliary head** (training only): predicts log(1 + detections) and log(1 + radiative power) on the target day; its loss has weight 0.2. It is not used at prediction time.

## Training (configs/final.yaml)
- AdamW, learning rate 0.0003, weight decay 0.01, batch size 4,096, binary cross-entropy on the hard label plus the auxiliary loss. At most 20 epochs.
- **Two stages per seed.** Stage 1 trains on the fit years before the last fit year and, after each epoch, scores the last fit year with the log loss of the probabilities corrected for the sampling of rows (average precision as tie-break); it stops after 2 epochs without improvement and keeps the best epoch count. Stage 2 retrains from the same seed on all fit rows for exactly that many epochs; its model is the one used.
- Settings (width, depth, dropout, learning rate, weight decay) were chosen among 8 random settings with one seed each on fold 2024 only and reused unchanged for fold 2023.
- Epochs chosen in stage 1 (seeds 0 to 7): fold 2024: 6, 3, 2, 5, 4, 6, 9, 5; fold 2023: 3, 4, 3, 3, 3, 7, 6, 4. About 5 to 18 minutes per seed on one 24 GB GPU (bfloat16); the training script runs on a CPU only for the synthetic sample.

## Ensemble
The 8 seeds (0 to 7) differ in initial weights and in batch order. The ensemble probability is the mean of the 8 seeds' probabilities. The ensemble is better than the average seed; the first 4 seeds alone give average precision 0.2888 on fold 2024 against 0.2913 for all 8.

## Calibration
Training uses a sample of rows (all fires, about 20 non-fire cell-days per fire), so raw probabilities are too high. Before log loss, Brier score, calibration error and thresholded scores, probabilities are corrected by multiplying the odds by the ratio of the true odds (the per-state fire rate of the real population, computed from the fit years) to the odds of the fit rows. The same correction, with the stopping year's own unthinned rows, gives the stopping score during training. No further recalibration is applied. A recalibration with isotonic regression fitted on fold 2024 was tried; it moved the 2024 Brier score by 0.00001, so it is not used. The calibration curves are in RESULTS.md. Calibration is good on fold 2024 (calibration error 0.00022) and worse on fold 2023 (0.00155, about twice the baseline's), where the probabilities run too low in the upper groups (the top tenth is predicted at 0.089 and observed at 0.099).

## Output
For every cell and day: one probability of a fire start on that day. Hazard maps draw these at the cell centres (`scripts/regenerate_results.py`, notebook 03).

## Weights
Trained weights are not committed. They are stored as training checkpoints (model weights and optimiser state; 3,713,718 bytes each; load them with `deep_multimodal.weights.load_weights`, which renames the stored keys) in the project results bucket under
`gs://<results-bucket>/proposed/dm/results/dmm_final_<year>_headline_<group>/ckpt/headline_seed<seed>_stage2.pt`
(`<year>` is the scored year 2024 or 2023, `<group>` is `a` for seeds 0 to 3 and `b` for seeds 4 to 7). The bucket name is not recorded here.

| Scored year | Seed | Storage group | SHA-256 of the weights file |
|---|---|---|---|
| 2023 | 0 | a | 6c9a5e0fb96568e6ca48fcf48ee97ea9b00a88dfbefce449a9ae82fe8d0928e1 |
| 2023 | 1 | a | 41465befd850cfdf5ce8f04cdb04f187de6ff4d0e7bd1a7e970bfcf448eb01f6 |
| 2023 | 2 | a | 0025a794ad1186def440fc8452e6682f39e690d2881452ea26742e6270a31332 |
| 2023 | 3 | a | 959217132733eb0ae1332e4dc9dc038d1e627bd3bb398053d2434cf7b4cd3d5d |
| 2023 | 4 | b | 3b2e8ce8c62c197d75aa15acb6f378fb6c739813b411801a48caa01d0d30ff6a |
| 2023 | 5 | b | 7324200b848d5ecc73cdc7544892f5c4c4318d1f45aec182484bd798d3fb82f2 |
| 2023 | 6 | b | c3238184a95a77a5b42eec3baf3c60f83cef85850245bba8779d283f3d624d5e |
| 2023 | 7 | b | cf0ccd723a9e2422108484bf482c9be26c034c464fbe53d476409d8585568827 |
| 2024 | 0 | a | 8bf9063e1a881acc440371e180ba9a37c1509b47297a5e0f0af641ab198a8584 |
| 2024 | 1 | a | 17c9394dfc06b9fb0e44f19ad837e511ac6942b166526569621506b9a01a8cec |
| 2024 | 2 | a | 6ef503f81f9e7eb509f771905edeec28305dd9c03d741a2d34952637c628f897 |
| 2024 | 3 | a | 28659f3cd1341d5c7a7588e2a72763dff49b5bbe1880fdd2e17d12175bcaa30b |
| 2024 | 4 | b | d7ec9c184ae094f8b73c625f47c57244d73379134fccde48dbf0a64cfd07cd16 |
| 2024 | 5 | b | 81de64df754567dff8fbf6da555715bc082a7fe133ca4f130d7bf0ec42753857 |
| 2024 | 6 | b | cadd80fa113508d848967809060952169b0c4b834cb61985c39c51d7b800a486 |
| 2024 | 7 | b | 31ea14acc177f2818603098966ff0150b3de2b68925b90e8bb97811adda96919 |

## How the port was checked against the research code
The network in this folder was written from the research code with new names. Checks run once against the research code (outside the test suite, because it needs that code and the weights):
- With each of the 16 trained checkpoints above loaded into both networks (names converted by `convert_legacy_state_dict`), the largest absolute difference in the logit on random inputs, including empty fire history and missing forecasts, was **0** (identical outputs). The parameter count, 296,587, equals the recorded count.
- With this folder's own data loading, standardising constants and network on **real 2024 rows** (every 145th scored row, 5,003 rows; seeds 0, 3 and 7), the logits agree with the logits the cloud run saved to within **0.04** (correlation 0.99998). The remaining difference is the rounding of the cloud run's bfloat16 arithmetic against float32 on the CPU.
- Training itself was not repeated (no new training run); the unit tests train on the synthetic sample.

## Limitations
- **Skill comes mainly from cells near existing fires.** Most of the model's average precision comes from rows with a recent detection in or near the cell. In the quiet stratum (no neighbouring detection in the previous 7 days) average precision is 0.064 on fold 2024 and 0.035 on fold 2023, against 0.29 and 0.25 over all rows. The model is much better at saying where fire continues than where a fire starts in a quiet area.
- **Quiet-area skill is weak**, and in the quiet stratum its ranking of days within a cell is slightly worse than the baseline's on both folds, and its thresholded detection on 2023 is lower (RESULTS.md).
- **The gain is not attributed to the architecture alone.** The model reads richer inputs than the baseline: the day-ahead forecast, 60 days of weather as a sequence, and the fire history of a 7 by 7 neighbourhood over 14 days. A boosted-tree model given the same inputs was not part of this comparison (and is not in this folder), so the improvement over the baseline cannot be split into inputs and architecture here. Do not read the table as an architecture result.
- **Calibration on fold 2023** is worse than the baseline's (calibration error 0.00155 against 0.00079).
- **Development folds only.** Both folds were used while building the model (the settings were chosen on 2024); no untouched test fold is reported. Average precision on sampled rows is not forecast skill in the operational sense; the Brier skill score against the cell climatology is small for both models (about 0.046 for this one).
- **Labels are detections**: a fire no satellite saw is not a label, a detection can be a burn that is not a wildfire, and there are no cause labels.
- **Coverage**: six states in central and eastern India, forest cells only, the fire season only (January to July rows are scored). It has not been tested elsewhere.
- **Forecast inputs** are 0.25 degree forecasts with known biases against the reanalysis (warm and dry by about 2 degrees and 10 points of humidity in this region); the model sees a constant offset, but a change of forecast system would shift them.
- **The grid files were reconstructed** after a deletion; maps use the reconstructed cell centres.
