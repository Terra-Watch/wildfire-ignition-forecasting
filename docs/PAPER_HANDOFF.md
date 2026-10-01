# Paper handoff: baseline results

Generated 2026-10-01 21:49 IST from the saved tables by `docs/methodology/make_paper_handoff.py`. Numbers below are read from files under `artifacts/results/`; the paper and the Word document were not touched.

## 0. Provenance
- Frozen baseline: local tag `baseline-final` on commit `5d7d83f` (not pushed); records in `artifacts/baseline_final/` (code hash, config snapshot, data manifest, artifact manifest).
- Code that produced each result (the `code_commit` stamped in the files): baseline tables `d55a8ca`, ablation tables `0f3ac9e`, sanity checks `d55a8ca`, fresh scoring `5d7d83f`. Rolling-origin folds 2023-2025 were fitted by `d55a8ca`, folds 2021 and 2022 by the earlier code `5af5654` (deviation j). Specification commit `99c867c`.
- Guard record (`guards.json`, code `9da492c`), every guard true: timestamp_before_day, shifted_feature_fails, identity_gate, label_shuffle, coordinate_unit_test, no_validation_information_in_tuning, power_gate, geo_ledger, provenance; label-shuffle mean AUC 0.5012.
- Data and artifact records: `wildfire verify-data` (21:2x on 2026-10-01) compared `data/MANIFEST.json` with the data folders and found no differences (exit code 0, no output). The frozen artifact manifest (`artifacts/baseline_final/artifact_manifest.json`) compared with the folders now: labels and features unchanged; results changed from 39 to 44 files, the five files added after the freeze (the fresh predictions, the fresh fit prevalences, the ledger and the two strata CSVs). The data manifest record itself was copied from the committed manifest.
- Tests at this commit: 169 passed in 12.81s. Sanity checks (`sanity.csv`): global label shuffle mean AP 0.0502 against prevalence 0.0522, cell x week shuffle max AP 0.0933 against climatology 0.1167, permuted holdout labels 0.0522: all passed.
- Fresh geographic test scored once: ledger entries AP, UK at 2026-10-01T20:38:14 (commit `5d7d83f`).

## 1. All 43 contrasts are INCONCLUSIVE
Every pre-declared contrast ends INCONCLUSIVE, because its minimum detectable effect (MDE 0.0127 to 0.0307) is above the power threshold of 0.010 (spec section 7: a verdict is INCONCLUSIVE when the MDE exceeds the threshold). This is the decision rule working as declared, not a null result: several gains are large and their 99% intervals are clear of zero. For example enriched boosting against tabular boosting is ΔAP +0.0352 with 99% CI 0.0251 to 0.0441, MDE 0.0127. The paper must not state a PASS for any contrast. The power gate is recorded as a guard (`power_gate`, a self-test of this rule); the saved contrasts obey it (none has an MDE at or below 0.010).

Every contrast (arm minus reference, AP, 99% paired ISO-week cluster bootstrap; "#n" is network seed n; worst state = state with the lowest paired ΔAP):

| contrast | arm | reference | delta_ap | ap_lo99 | ap_hi99 | mde | worst_state | worst_state_delta_ap | verdict |
|---|---|---|---|---|---|---|---|---|---|
| enriched_boosting_vs_tabular_boosting | boosting@enriched | boosting@tabular | 0.0352 | 0.0251 | 0.0441 | 0.0127 | JH | 0.0074 | INCONCLUSIVE |
| enriched_boosting_vs_enriched_logistic | boosting@enriched | logistic_with_burn@enriched | 0.0623 | 0.0433 | 0.0812 | 0.0254 | JH | 0.0380 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#0 | boosting@enriched | -0.0253 | -0.0457 | -0.0068 | 0.0263 | MP | -0.0339 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#1 | boosting@enriched | -0.0266 | -0.0487 | -0.0076 | 0.0275 | TS | -0.0396 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#2 | boosting@enriched | -0.0291 | -0.0501 | -0.0105 | 0.0264 | TS | -0.0382 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#3 | boosting@enriched | -0.0279 | -0.0543 | -0.0072 | 0.0302 | JH | -0.0423 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#4 | boosting@enriched | -0.0297 | -0.0521 | -0.0105 | 0.0275 | TS | -0.0424 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#5 | boosting@enriched | -0.0325 | -0.0572 | -0.0116 | 0.0307 | TS | -0.0473 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#6 | boosting@enriched | -0.0269 | -0.0511 | -0.0080 | 0.0282 | TS | -0.0409 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#7 | boosting@enriched | -0.0305 | -0.0537 | -0.0109 | 0.0289 | JH | -0.0419 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#8 | boosting@enriched | -0.0245 | -0.0463 | -0.0065 | 0.0263 | TS | -0.0320 | INCONCLUSIVE |
| network_vs_enriched_boosting | tabular_network@enriched#9 | boosting@enriched | -0.0262 | -0.0486 | -0.0079 | 0.0270 | TS | -0.0402 | INCONCLUSIVE |
| landcover_season_vs_climatology | landcover_season | climatology_cell | -0.0269 | -0.0402 | -0.0129 | 0.0176 | MH | -0.0326 | INCONCLUSIVE |
| fwi_season_vs_climatology | fwi_season | climatology_cell | -0.0100 | -0.0238 | 0.0028 | 0.0176 | MH | -0.0275 | INCONCLUSIVE |
| logistic_weather_static@tabular_vs_climatology | logistic_weather_static@tabular | climatology_cell | 0.0025 | -0.0111 | 0.0148 | 0.0170 | MH | -0.0207 | INCONCLUSIVE |
| logistic_weather_static@enriched_vs_climatology | logistic_weather_static@enriched | climatology_cell | 0.0683 | 0.0513 | 0.0820 | 0.0198 | MH | 0.0366 | INCONCLUSIVE |
| logistic_with_burn@tabular_vs_climatology | logistic_with_burn@tabular | climatology_cell | 0.0086 | -0.0050 | 0.0214 | 0.0168 | MH | -0.0126 | INCONCLUSIVE |
| logistic_with_burn@enriched_vs_climatology | logistic_with_burn@enriched | climatology_cell | 0.0686 | 0.0516 | 0.0825 | 0.0199 | MH | 0.0372 | INCONCLUSIVE |
| boosting@tabular_vs_climatology | boosting@tabular | climatology_cell | 0.0958 | 0.0718 | 0.1160 | 0.0296 | MH | 0.0621 | INCONCLUSIVE |
| boosting@enriched_vs_climatology | boosting@enriched | climatology_cell | 0.1310 | 0.1064 | 0.1509 | 0.0307 | MH | 0.0898 | INCONCLUSIVE |
| tabular_network@tabular#0_vs_climatology | tabular_network@tabular#0 | climatology_cell | 0.0766 | 0.0596 | 0.0856 | 0.0174 | MH | 0.0484 | INCONCLUSIVE |
| tabular_network@tabular#1_vs_climatology | tabular_network@tabular#1 | climatology_cell | 0.0749 | 0.0578 | 0.0840 | 0.0181 | MH | 0.0506 | INCONCLUSIVE |
| tabular_network@tabular#2_vs_climatology | tabular_network@tabular#2 | climatology_cell | 0.0744 | 0.0570 | 0.0835 | 0.0182 | MH | 0.0493 | INCONCLUSIVE |
| tabular_network@tabular#3_vs_climatology | tabular_network@tabular#3 | climatology_cell | 0.0735 | 0.0561 | 0.0825 | 0.0179 | MH | 0.0480 | INCONCLUSIVE |
| tabular_network@tabular#4_vs_climatology | tabular_network@tabular#4 | climatology_cell | 0.0738 | 0.0564 | 0.0836 | 0.0183 | MH | 0.0468 | INCONCLUSIVE |
| tabular_network@tabular#5_vs_climatology | tabular_network@tabular#5 | climatology_cell | 0.0728 | 0.0565 | 0.0816 | 0.0172 | MH | 0.0466 | INCONCLUSIVE |
| tabular_network@tabular#6_vs_climatology | tabular_network@tabular#6 | climatology_cell | 0.0744 | 0.0581 | 0.0835 | 0.0175 | MH | 0.0465 | INCONCLUSIVE |
| tabular_network@tabular#7_vs_climatology | tabular_network@tabular#7 | climatology_cell | 0.0728 | 0.0563 | 0.0830 | 0.0181 | MH | 0.0484 | INCONCLUSIVE |
| tabular_network@tabular#8_vs_climatology | tabular_network@tabular#8 | climatology_cell | 0.0734 | 0.0562 | 0.0832 | 0.0183 | MH | 0.0458 | INCONCLUSIVE |
| tabular_network@tabular#9_vs_climatology | tabular_network@tabular#9 | climatology_cell | 0.0759 | 0.0594 | 0.0849 | 0.0176 | MH | 0.0490 | INCONCLUSIVE |
| tabular_network@enriched#0_vs_climatology | tabular_network@enriched#0 | climatology_cell | 0.1057 | 0.0840 | 0.1187 | 0.0245 | MH | 0.0710 | INCONCLUSIVE |
| tabular_network@enriched#1_vs_climatology | tabular_network@enriched#1 | climatology_cell | 0.1044 | 0.0821 | 0.1167 | 0.0242 | MH | 0.0689 | INCONCLUSIVE |
| tabular_network@enriched#2_vs_climatology | tabular_network@enriched#2 | climatology_cell | 0.1019 | 0.0793 | 0.1137 | 0.0242 | MH | 0.0675 | INCONCLUSIVE |
| tabular_network@enriched#3_vs_climatology | tabular_network@enriched#3 | climatology_cell | 0.1031 | 0.0793 | 0.1183 | 0.0263 | MH | 0.0693 | INCONCLUSIVE |
| tabular_network@enriched#4_vs_climatology | tabular_network@enriched#4 | climatology_cell | 0.1013 | 0.0798 | 0.1132 | 0.0235 | MH | 0.0655 | INCONCLUSIVE |
| tabular_network@enriched#5_vs_climatology | tabular_network@enriched#5 | climatology_cell | 0.0985 | 0.0754 | 0.1122 | 0.0256 | MH | 0.0635 | INCONCLUSIVE |
| tabular_network@enriched#6_vs_climatology | tabular_network@enriched#6 | climatology_cell | 0.1040 | 0.0830 | 0.1161 | 0.0236 | MH | 0.0663 | INCONCLUSIVE |
| tabular_network@enriched#7_vs_climatology | tabular_network@enriched#7 | climatology_cell | 0.1004 | 0.0771 | 0.1144 | 0.0259 | MH | 0.0671 | INCONCLUSIVE |
| tabular_network@enriched#8_vs_climatology | tabular_network@enriched#8 | climatology_cell | 0.1065 | 0.0846 | 0.1189 | 0.0240 | MH | 0.0670 | INCONCLUSIVE |
| tabular_network@enriched#9_vs_climatology | tabular_network@enriched#9 | climatology_cell | 0.1047 | 0.0833 | 0.1163 | 0.0230 | MH | 0.0658 | INCONCLUSIVE |
| coords_only_vs_climatology | coords_only | climatology_cell | -0.0085 | -0.0228 | 0.0043 | 0.0185 | TS | -0.0254 | INCONCLUSIVE |
| cell_id_only_vs_climatology | cell_id_only | climatology_cell | 0.0056 | -0.0096 | 0.0161 | 0.0179 | TS | -0.0183 | INCONCLUSIVE |
| static_only_vs_climatology | static_only | climatology_cell | -0.0055 | -0.0216 | 0.0074 | 0.0195 | TS | -0.0252 | INCONCLUSIVE |

## 2. Main table: enriched level
Pooled over validation years 2021-2025 (2025 is a development fold, not untouched), 4,521,002 rows and 240,495 onsets. Log loss and Brier use design weights and the per-state true rate; AP and both AUCs use the raw scores. The network row is the mean of 10 seeds (seed standard deviation of AP 0.0025, spatial AUC 0.0030, temporal AUC 0.0015). The three arms without levels and the three diagnostic arms are listed for reference; the diagnostics measure the place shortcut and are not gated.

| arm | spatial AUC | temporal AUC | AP | log loss | Brier | capture 5% | lift 5% |
|---|---|---|---|---|---|---|---|
| climatology_cell | 0.6537 | 0.5909 | 0.0951 | 0.0889 | 0.0179 | 0.1435 | 2.8698 |
| landcover_season | 0.5532 | 0.5557 | 0.0682 | 0.0907 | 0.0180 | 0.0845 | 1.6899 |
| fwi_season | 0.5536 | 0.6441 | 0.0851 | 0.0886 | 0.0180 | 0.1093 | 2.1851 |
| logistic_weather_static@enriched | 0.7332 | 0.7469 | 0.1634 | 0.0818 | 0.0177 | 0.2050 | 4.0990 |
| logistic_with_burn@enriched | 0.7338 | 0.7465 | 0.1637 | 0.0818 | 0.0177 | 0.2054 | 4.1079 |
| tabular_network@enriched | 0.7692 | 0.7798 | 0.1982 | 0.0799 | 0.0174 | 0.2380 | 4.7591 |
| boosting@enriched | 0.8110 | 0.7896 | 0.2261 | 0.0768 | 0.0172 | 0.2490 | 4.9797 |
| coords_only | 0.6960 | 0.4865 | 0.0866 | 0.0906 | 0.0180 | 0.1123 | 2.2455 |
| cell_id_only | 0.7115 | 0.4676 | 0.1007 | 0.1048 | 0.0180 | 0.1223 | 2.4460 |
| static_only | 0.6870 | 0.4864 | 0.0896 | 0.0906 | 0.0180 | 0.1131 | 2.2615 |

## 3. Tabular versus enriched
| arm | spatial AUC tabular | temporal AUC tabular | AP tabular | spatial AUC enriched | temporal AUC enriched | AP enriched | AP gain |
|---|---|---|---|---|---|---|---|
| logistic_weather_static | 0.6102 | 0.6783 | 0.0976 | 0.7332 | 0.7469 | 0.1634 | 0.0658 |
| logistic_with_burn | 0.6300 | 0.6745 | 0.1038 | 0.7338 | 0.7465 | 0.1637 | 0.0600 |
| boosting | 0.7688 | 0.7562 | 0.1909 | 0.8110 | 0.7896 | 0.2261 | 0.0352 |
| tabular_network | 0.7393 | 0.7676 | 0.1694 | 0.7692 | 0.7798 | 0.1982 | 0.0288 |

The information comparison (enriched boosting against tabular boosting) is contrast (a): ΔAP +0.0352, 99% CI 0.0251 to 0.0441, MDE 0.0127, INCONCLUSIVE. Ablations (secondary, post hoc, `docs/decisions/2026-10-01-ablations.md`; boosting@enriched with the same folds and settings; ΔAP against boosting@enriched, AP 0.2261):

| contrast | delta_ap | ap_lo99 | ap_hi99 | mde | verdict |
|---|---|---|---|---|---|
| A_no_soil_anomaly_vs_boosting@enriched | 0.0010 | -0.0003 | 0.0025 | 0.0018 | FAIL |
| B_windows_only_vs_boosting@enriched | -0.0384 | -0.0480 | -0.0300 | 0.0123 | INCONCLUSIVE |
| B_neighbourhood_only_vs_boosting@enriched | 0.0032 | -0.0033 | 0.0106 | 0.0095 | INCONCLUSIVE |

(A_no_soil_anomaly removes `sw1_anom` and `sw2_anom`; B_windows_only keeps the weather windows and their anomalies but no neighbourhood block; B_neighbourhood_only keeps the neighbourhood block but no windows. Verdict labels here are descriptive only.) Post hoc stratification by recent neighbour onsets (`docs/decisions/2026-10-01-onsets-recent-strata.md`): the enriched AP gain is +0.0294 (99% CI 0.0202 to 0.0372) in rows with recent onsets in the 8 neighbouring cells (94.1% of positives) and +0.0034 (99% CI -0.0057 to 0.0170) in rows with none, so a gain there is not demonstrated.

## 4. Rolling-origin results per year
Each year is scored by models fitted on all earlier years from 2019, January to July rows only. Rows and onsets per year: 2021: 1,034,752 / 57,080, 2022: 839,785 / 44,058, 2023: 939,632 / 50,420, 2024: 726,279 / 37,679, 2025: 980,554 / 51,258. Network rows are means over 10 seeds.

AP:

| arm | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| climatology_cell | 0.0704 | 0.1002 | 0.0844 | 0.1059 | 0.1167 |
| logistic_with_burn@tabular | 0.1105 | 0.1323 | 0.1010 | 0.1270 | 0.1122 |
| logistic_with_burn@enriched | 0.1685 | 0.1818 | 0.1590 | 0.1788 | 0.1867 |
| boosting@tabular | 0.1736 | 0.2473 | 0.2107 | 0.1988 | 0.2000 |
| boosting@enriched | 0.2075 | 0.2693 | 0.2366 | 0.2471 | 0.2461 |
| tabular_network@tabular | 0.1511 | 0.1674 | 0.1663 | 0.1768 | 0.1908 |
| tabular_network@enriched | 0.1653 | 0.1885 | 0.2039 | 0.2273 | 0.2323 |

Within-day spatial AUC:

| arm | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| climatology_cell | 0.6003 | 0.6480 | 0.6399 | 0.6826 | 0.7001 |
| logistic_with_burn@tabular | 0.6276 | 0.6413 | 0.6041 | 0.6368 | 0.6410 |
| logistic_with_burn@enriched | 0.7192 | 0.7355 | 0.7310 | 0.7425 | 0.7412 |
| boosting@tabular | 0.7463 | 0.7845 | 0.7629 | 0.7748 | 0.7759 |
| boosting@enriched | 0.7939 | 0.8186 | 0.8042 | 0.8169 | 0.8217 |
| tabular_network@tabular | 0.7008 | 0.7412 | 0.7346 | 0.7527 | 0.7684 |
| tabular_network@enriched | 0.6903 | 0.7774 | 0.7805 | 0.7981 | 0.8015 |

Within-cell temporal AUC:

| arm | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| climatology_cell | 0.5386 | 0.6409 | 0.5975 | 0.6265 | 0.6786 |
| logistic_with_burn@tabular | 0.7321 | 0.7804 | 0.6883 | 0.7352 | 0.7380 |
| logistic_with_burn@enriched | 0.7726 | 0.7853 | 0.7348 | 0.7701 | 0.7881 |
| boosting@tabular | 0.7749 | 0.8156 | 0.7711 | 0.7724 | 0.8055 |
| boosting@enriched | 0.8056 | 0.8185 | 0.7776 | 0.7958 | 0.8237 |
| tabular_network@tabular | 0.7715 | 0.7929 | 0.7563 | 0.7682 | 0.8058 |
| tabular_network@enriched | 0.7623 | 0.7807 | 0.7726 | 0.7911 | 0.8172 |

## 5. Fresh states (UK and AP), scored once
Every arm fitted on all years before 2026 (2,000,000-row cap) and scored on the fresh rows: AP 782,694 rows (37,676 onsets), UK 590,870 rows (27,511 onsets), years 2020-2025, every fire-season month through 2025-07-31.

**What the specification declares for the fresh test rows.** `docs/baseline_spec.md` section 2: "fresh geographic test: UK always; AP, KA, TN, KL only if their fire data passes the completeness check | once, every access recorded in a ledger". `configs/baseline.yaml` `fresh_geographic_test`: `states_always: [UK]`, `states_if_fire_data_complete: [AP, KA, TN, KL]`, `scores_allowed: 1`, `ledger: geo_test_ledger.csv`. Neither names a month or row set. The January-to-July rule is declared for validation years: "Every validation year uses the rows of January to July (the 2025 data stops at the end of July)" (spec, "Compute limits declared in advance"; `validation_months: [1, 2, 3, 4, 5, 6, 7]`, commented "rows used to validate each year"). So the all-months scoring below is not a deviation from a declared fresh-test row set; it differs from the validation-year rule and from the earlier reference run (deviation o). Because the spec fixes no months for the fresh test, the table scored as run (all months) is given first and the January-July table, which applies the validation-year rule by analogy and was computed after the results were known, is labelled a post hoc sensitivity.

Unit design weights and the configured population rate 0.0147 are used, so log loss is less comparable than AP and the AUC views. Lift is AP divided by the state's sampled prevalence. No state-level contrast tests were declared for the fresh states; no verdicts are given. Networks are mean ± SD over the 10 seeds with the per-seed range [min-max]; every other arm has one fit. The figures AP 0.1144 (AP) and 0.1238 (UK), quoted in earlier discussion, are the AP of **`tabular_network@enriched`** (mean over 10 seeds); the tabular-level network has AP 0.2074 (AP) and 0.0813 (UK).

### 5a. UK and AP, all scored months (as scored)
| state | rows | onsets | prevalence | arm | seeds | AP | lift (AP / prevalence) | AUC pooled | AUC spatial | AUC temporal | log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UK | 590870 | 27511 | 0.0466 | climatology_cell | 1 | 0.0564 | 1.2110 | 0.5642 | 0.5917 | 0.5194 | 0.2120 |
| UK | 590870 | 27511 | 0.0466 | landcover_season | 1 | 0.0540 | 1.1594 | 0.5538 | 0.4540 | 0.5796 | 0.2133 |
| UK | 590870 | 27511 | 0.0466 | fwi_season | 1 | 0.0869 | 1.8672 | 0.6197 | 0.5445 | 0.6196 | 0.2236 |
| UK | 590870 | 27511 | 0.0466 | logistic_weather_static@tabular | 1 | 0.1008 | 2.1642 | 0.6734 | 0.5803 | 0.6587 | 0.2187 |
| UK | 590870 | 27511 | 0.0466 | logistic_weather_static@enriched | 1 | 0.1267 | 2.7221 | 0.7110 | 0.6583 | 0.6797 | 0.2134 |
| UK | 590870 | 27511 | 0.0466 | logistic_with_burn@tabular | 1 | 0.1077 | 2.3134 | 0.6805 | 0.5932 | 0.6568 | 0.2032 |
| UK | 590870 | 27511 | 0.0466 | logistic_with_burn@enriched | 1 | 0.1265 | 2.7170 | 0.7110 | 0.6592 | 0.6789 | 0.2145 |
| UK | 590870 | 27511 | 0.0466 | boosting@tabular | 1 | 0.1075 | 2.3083 | 0.6642 | 0.6316 | 0.6477 | 0.2208 |
| UK | 590870 | 27511 | 0.0466 | boosting@enriched | 1 | 0.0979 | 2.1032 | 0.7185 | 0.6562 | 0.7211 | 0.2025 |
| UK | 590870 | 27511 | 0.0466 | tabular_network@tabular | 10 | 0.0813 ± 0.0038 [0.0738-0.0874] | 1.7469 ± 0.0823 [1.5858-1.8773] | 0.6376 ± 0.0077 [0.6270-0.6544] | 0.6141 ± 0.0040 [0.6082-0.6206] | 0.6161 ± 0.0091 [0.6057-0.6356] | 0.2861 ± 0.0128 [0.2717-0.3175] |
| UK | 590870 | 27511 | 0.0466 | tabular_network@enriched | 10 | 0.1238 ± 0.0078 [0.1036-0.1313] | 2.6579 ± 0.1669 [2.2258-2.8201] | 0.7308 ± 0.0160 [0.6972-0.7508] | 0.6912 ± 0.0075 [0.6745-0.7012] | 0.7066 ± 0.0260 [0.6534-0.7472] | 0.2313 ± 0.0077 [0.2166-0.2415] |
| UK | 590870 | 27511 | 0.0466 | coords_only | 1 | 0.0458 | 0.9840 | 0.4972 | 0.5160 | 0.5000 | 0.2362 |
| UK | 590870 | 27511 | 0.0466 | cell_id_only | 1 | 0.0466 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.2106 |
| UK | 590870 | 27511 | 0.0466 | static_only | 1 | 0.0509 | 1.0926 | 0.5404 | 0.5128 | 0.5000 | 0.2073 |
| AP | 782694 | 37676 | 0.0481 | climatology_cell | 1 | 0.0887 | 1.8423 | 0.6417 | 0.6139 | 0.5693 | 0.2083 |
| AP | 782694 | 37676 | 0.0481 | landcover_season | 1 | 0.0772 | 1.6039 | 0.6503 | 0.5276 | 0.6119 | 0.2154 |
| AP | 782694 | 37676 | 0.0481 | fwi_season | 1 | 0.0961 | 1.9970 | 0.6674 | 0.5549 | 0.6453 | 0.2156 |
| AP | 782694 | 37676 | 0.0481 | logistic_weather_static@tabular | 1 | 0.1050 | 2.1815 | 0.7055 | 0.5701 | 0.6747 | 0.2051 |
| AP | 782694 | 37676 | 0.0481 | logistic_weather_static@enriched | 1 | 0.1456 | 3.0246 | 0.7565 | 0.6521 | 0.7206 | 0.1972 |
| AP | 782694 | 37676 | 0.0481 | logistic_with_burn@tabular | 1 | 0.1061 | 2.2037 | 0.7071 | 0.5879 | 0.6738 | 0.2064 |
| AP | 782694 | 37676 | 0.0481 | logistic_with_burn@enriched | 1 | 0.1453 | 3.0187 | 0.7567 | 0.6527 | 0.7202 | 0.1972 |
| AP | 782694 | 37676 | 0.0481 | boosting@tabular | 1 | 0.2494 | 5.1808 | 0.7742 | 0.6558 | 0.7355 | 0.1921 |
| AP | 782694 | 37676 | 0.0481 | boosting@enriched | 1 | 0.2945 | 6.1186 | 0.8147 | 0.7188 | 0.7600 | 0.1815 |
| AP | 782694 | 37676 | 0.0481 | tabular_network@tabular | 10 | 0.2074 ± 0.0092 [0.1925-0.2195] | 4.3084 ± 0.1916 [3.9993-4.5601] | 0.7527 ± 0.0055 [0.7445-0.7606] | 0.6403 ± 0.0042 [0.6338-0.6457] | 0.7319 ± 0.0021 [0.7285-0.7357] | 0.1946 ± 0.0019 [0.1918-0.1976] |
| AP | 782694 | 37676 | 0.0481 | tabular_network@enriched | 10 | 0.1144 ± 0.0071 [0.1061-0.1304] | 2.3765 ± 0.1478 [2.2034-2.7094] | 0.7507 ± 0.0069 [0.7427-0.7643] | 0.6675 ± 0.0042 [0.6631-0.6757] | 0.7101 ± 0.0055 [0.7017-0.7219] | 0.2558 ± 0.0182 [0.2325-0.2842] |
| AP | 782694 | 37676 | 0.0481 | coords_only | 1 | 0.0496 | 1.0309 | 0.5021 | 0.5120 | 0.5000 | 0.2242 |
| AP | 782694 | 37676 | 0.0481 | cell_id_only | 1 | 0.0481 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.2172 |
| AP | 782694 | 37676 | 0.0481 | static_only | 1 | 0.0602 | 1.2515 | 0.5802 | 0.5548 | 0.5000 | 0.2204 |

### 5b. UK, all scored months (as scored): the UK rows of 5a repeated beside 5c
| state | rows | onsets | prevalence | arm | seeds | AP | lift (AP / prevalence) | AUC pooled | AUC spatial | AUC temporal | log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UK | 590870 | 27511 | 0.0466 | climatology_cell | 1 | 0.0564 | 1.2110 | 0.5642 | 0.5917 | 0.5194 | 0.2120 |
| UK | 590870 | 27511 | 0.0466 | landcover_season | 1 | 0.0540 | 1.1594 | 0.5538 | 0.4540 | 0.5796 | 0.2133 |
| UK | 590870 | 27511 | 0.0466 | fwi_season | 1 | 0.0869 | 1.8672 | 0.6197 | 0.5445 | 0.6196 | 0.2236 |
| UK | 590870 | 27511 | 0.0466 | logistic_weather_static@tabular | 1 | 0.1008 | 2.1642 | 0.6734 | 0.5803 | 0.6587 | 0.2187 |
| UK | 590870 | 27511 | 0.0466 | logistic_weather_static@enriched | 1 | 0.1267 | 2.7221 | 0.7110 | 0.6583 | 0.6797 | 0.2134 |
| UK | 590870 | 27511 | 0.0466 | logistic_with_burn@tabular | 1 | 0.1077 | 2.3134 | 0.6805 | 0.5932 | 0.6568 | 0.2032 |
| UK | 590870 | 27511 | 0.0466 | logistic_with_burn@enriched | 1 | 0.1265 | 2.7170 | 0.7110 | 0.6592 | 0.6789 | 0.2145 |
| UK | 590870 | 27511 | 0.0466 | boosting@tabular | 1 | 0.1075 | 2.3083 | 0.6642 | 0.6316 | 0.6477 | 0.2208 |
| UK | 590870 | 27511 | 0.0466 | boosting@enriched | 1 | 0.0979 | 2.1032 | 0.7185 | 0.6562 | 0.7211 | 0.2025 |
| UK | 590870 | 27511 | 0.0466 | tabular_network@tabular | 10 | 0.0813 ± 0.0038 [0.0738-0.0874] | 1.7469 ± 0.0823 [1.5858-1.8773] | 0.6376 ± 0.0077 [0.6270-0.6544] | 0.6141 ± 0.0040 [0.6082-0.6206] | 0.6161 ± 0.0091 [0.6057-0.6356] | 0.2861 ± 0.0128 [0.2717-0.3175] |
| UK | 590870 | 27511 | 0.0466 | tabular_network@enriched | 10 | 0.1238 ± 0.0078 [0.1036-0.1313] | 2.6579 ± 0.1669 [2.2258-2.8201] | 0.7308 ± 0.0160 [0.6972-0.7508] | 0.6912 ± 0.0075 [0.6745-0.7012] | 0.7066 ± 0.0260 [0.6534-0.7472] | 0.2313 ± 0.0077 [0.2166-0.2415] |
| UK | 590870 | 27511 | 0.0466 | coords_only | 1 | 0.0458 | 0.9840 | 0.4972 | 0.5160 | 0.5000 | 0.2362 |
| UK | 590870 | 27511 | 0.0466 | cell_id_only | 1 | 0.0466 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.2106 |
| UK | 590870 | 27511 | 0.0466 | static_only | 1 | 0.0509 | 1.0926 | 0.5404 | 0.5128 | 0.5000 | 0.2073 |

### 5c. UK, January to July only: SENSITIVITY (POST HOC)
Rows of months 1-7 of the saved UK predictions, scored with the same metric code; nothing was refitted or re-scored. Fitted models are those of the fresh scoring.

| state | rows | onsets | prevalence | arm | seeds | AP | lift (AP / prevalence) | AUC pooled | AUC spatial | AUC temporal | log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UK | 508703 | 22970 | 0.0452 | climatology_cell | 1 | 0.0539 | 1.1934 | 0.5659 | 0.5872 | 0.5169 | 0.2057 |
| UK | 508703 | 22970 | 0.0452 | landcover_season | 1 | 0.0558 | 1.2360 | 0.5992 | 0.4530 | 0.6191 | 0.1979 |
| UK | 508703 | 22970 | 0.0452 | fwi_season | 1 | 0.0947 | 2.0972 | 0.6789 | 0.5427 | 0.6597 | 0.2102 |
| UK | 508703 | 22970 | 0.0452 | logistic_weather_static@tabular | 1 | 0.1080 | 2.3926 | 0.7079 | 0.5929 | 0.6743 | 0.2014 |
| UK | 508703 | 22970 | 0.0452 | logistic_weather_static@enriched | 1 | 0.1285 | 2.8463 | 0.7159 | 0.6613 | 0.6639 | 0.2020 |
| UK | 508703 | 22970 | 0.0452 | logistic_with_burn@tabular | 1 | 0.1160 | 2.5691 | 0.7154 | 0.6096 | 0.6738 | 0.1866 |
| UK | 508703 | 22970 | 0.0452 | logistic_with_burn@enriched | 1 | 0.1282 | 2.8399 | 0.7159 | 0.6622 | 0.6633 | 0.2030 |
| UK | 508703 | 22970 | 0.0452 | boosting@tabular | 1 | 0.0801 | 1.7736 | 0.6574 | 0.6285 | 0.6250 | 0.2144 |
| UK | 508703 | 22970 | 0.0452 | boosting@enriched | 1 | 0.0814 | 1.8017 | 0.7078 | 0.6381 | 0.7199 | 0.2029 |
| UK | 508703 | 22970 | 0.0452 | tabular_network@tabular | 10 | 0.0689 ± 0.0034 [0.0635-0.0759] | 1.5259 ± 0.0752 [1.4066-1.6813] | 0.6366 ± 0.0091 [0.6255-0.6579] | 0.6214 ± 0.0036 [0.6161-0.6270] | 0.5937 ± 0.0107 [0.5798-0.6180] | 0.2740 ± 0.0126 [0.2608-0.3051] |
| UK | 508703 | 22970 | 0.0452 | tabular_network@enriched | 10 | 0.1072 ± 0.0048 [0.0959-0.1116] | 2.3750 ± 0.1072 [2.1246-2.4711] | 0.7303 ± 0.0135 [0.7001-0.7456] | 0.6929 ± 0.0061 [0.6783-0.6999] | 0.6964 ± 0.0235 [0.6456-0.7267] | 0.2222 ± 0.0069 [0.2086-0.2300] |
| UK | 508703 | 22970 | 0.0452 | coords_only | 1 | 0.0435 | 0.9644 | 0.4881 | 0.5037 | 0.5000 | 0.2297 |
| UK | 508703 | 22970 | 0.0452 | cell_id_only | 1 | 0.0452 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.2047 |
| UK | 508703 | 22970 | 0.0452 | static_only | 1 | 0.0505 | 1.1187 | 0.5507 | 0.5225 | 0.5000 | 0.2010 |

### 5d. Onset counts per month, pooled 2020-2025 (scored rows)
| month | UK rows | UK onsets | UK onset share of rows | AP rows | AP onsets | AP onset share of rows |
|---|---|---|---|---|---|---|
| 1 | 90093 | 3002 | 0.0333 | 137849 | 2160 | 0.0157 |
| 2 | 82266 | 3513 | 0.0427 | 125647 | 5045 | 0.0402 |
| 3 | 88131 | 4619 | 0.0524 | 133508 | 11812 | 0.0885 |
| 4 | 80236 | 7358 | 0.0917 | 128937 | 9870 | 0.0765 |
| 5 | 86649 | 2589 | 0.0299 | 134751 | 4334 | 0.0322 |
| 6 | 81309 | 1870 | 0.0230 | 1213 | 1213 | 1.0000 |
| 7 | 19 | 19 | 1.0000 | 201 | 201 | 1.0000 |
| 8 | 8 | 8 | 1.0000 | 271 | 271 | 1.0000 |
| 9 | 18 | 18 | 1.0000 | 261 | 261 | 1.0000 |
| 10 | 672 | 672 | 1.0000 | 527 | 527 | 1.0000 |
| 11 | 1250 | 1250 | 1.0000 | 607 | 607 | 1.0000 |
| 12 | 80219 | 2593 | 0.0323 | 118922 | 1375 | 0.0116 |
| total | 590870 | 27511 | 0.0466 | 782694 | 37676 | 0.0481 |

In months where the share is 1.0 every scored row is an onset: the matched non-fire days are drawn only inside the state's fire season, while onsets can fall in any month. The scored rows of those months therefore carry positives only (UK: months 7 to 11, 1,967 rows; AP: months 6 to 11, 3,080 rows). This affects the sampled prevalence, the pooled AUC and AP of the as-scored tables, and it is part of the difference between 5a and 5c.

### 5e. Per-seed range of the network, UK and AP
| state | level | AP min | AP max | AP mean | AP SD | AUC pooled min | AUC pooled max | AUC pooled mean | AUC pooled SD | AUC spatial min | AUC spatial max | AUC spatial mean | AUC spatial SD | AUC temporal min | AUC temporal max | AUC temporal mean | AUC temporal SD | log loss min | log loss max | log loss mean | log loss SD | AP by seed 0-9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UK | tabular | 0.0738 | 0.0874 | 0.0813 | 0.0038 | 0.6270 | 0.6544 | 0.6376 | 0.0077 | 0.6082 | 0.6206 | 0.6141 | 0.0040 | 0.6057 | 0.6356 | 0.6161 | 0.0091 | 0.2717 | 0.3175 | 0.2861 | 0.0128 | 0.0738, 0.0799, 0.0771, 0.0831, 0.0874, 0.0830, 0.0843, 0.0832, 0.0807, 0.0808 |
| UK | enriched | 0.1036 | 0.1313 | 0.1238 | 0.0078 | 0.6972 | 0.7508 | 0.7308 | 0.0160 | 0.6745 | 0.7012 | 0.6912 | 0.0075 | 0.6534 | 0.7472 | 0.7066 | 0.0260 | 0.2166 | 0.2415 | 0.2313 | 0.0077 | 0.1259, 0.1232, 0.1036, 0.1264, 0.1313, 0.1252, 0.1202, 0.1299, 0.1279, 0.1239 |
| AP | tabular | 0.1925 | 0.2195 | 0.2074 | 0.0092 | 0.7445 | 0.7606 | 0.7527 | 0.0055 | 0.6338 | 0.6457 | 0.6403 | 0.0042 | 0.7285 | 0.7357 | 0.7319 | 0.0021 | 0.1918 | 0.1976 | 0.1946 | 0.0019 | 0.2070, 0.2015, 0.2105, 0.2173, 0.2131, 0.2195, 0.1989, 0.1981, 0.2154, 0.1925 |
| AP | enriched | 0.1061 | 0.1304 | 0.1144 | 0.0071 | 0.7427 | 0.7643 | 0.7507 | 0.0069 | 0.6631 | 0.6757 | 0.6675 | 0.0042 | 0.7017 | 0.7219 | 0.7101 | 0.0055 | 0.2325 | 0.2842 | 0.2558 | 0.0182 | 0.1138, 0.1134, 0.1061, 0.1065, 0.1195, 0.1137, 0.1093, 0.1304, 0.1179, 0.1133 |

### 5f. AP, January to July only: SENSITIVITY (POST HOC)
Rows of months 1-7 of the saved AP predictions, same metric code; nothing refitted or re-scored. Onset-only rows (rows in a month where the state has no matched non-fire rows, so every row is an onset) inside each state's January-July set: **UK 19** of 508,703 rows (month 7), **AP 1,414** of 662,106 rows (months 6 and 7).

| state | rows | onsets | prevalence | arm | seeds | AP | lift (AP / prevalence) | AUC pooled | AUC spatial | AUC temporal | log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP | 662106 | 34635 | 0.0523 | climatology_cell | 1 | 0.0934 | 1.7846 | 0.6421 | 0.6140 | 0.5595 | 0.2240 |
| AP | 662106 | 34635 | 0.0523 | landcover_season | 1 | 0.0799 | 1.5281 | 0.6485 | 0.5345 | 0.6092 | 0.2275 |
| AP | 662106 | 34635 | 0.0523 | fwi_season | 1 | 0.1002 | 1.9155 | 0.6673 | 0.5457 | 0.6515 | 0.2282 |
| AP | 662106 | 34635 | 0.0523 | logistic_weather_static@tabular | 1 | 0.1088 | 2.0805 | 0.7015 | 0.5784 | 0.6658 | 0.2187 |
| AP | 662106 | 34635 | 0.0523 | logistic_weather_static@enriched | 1 | 0.1730 | 3.3063 | 0.7590 | 0.6676 | 0.7067 | 0.2117 |
| AP | 662106 | 34635 | 0.0523 | logistic_with_burn@tabular | 1 | 0.1101 | 2.1049 | 0.7031 | 0.5934 | 0.6645 | 0.2200 |
| AP | 662106 | 34635 | 0.0523 | logistic_with_burn@enriched | 1 | 0.1737 | 3.3210 | 0.7595 | 0.6684 | 0.7065 | 0.2116 |
| AP | 662106 | 34635 | 0.0523 | boosting@tabular | 1 | 0.1991 | 3.8062 | 0.7504 | 0.6449 | 0.6956 | 0.2146 |
| AP | 662106 | 34635 | 0.0523 | boosting@enriched | 1 | 0.2572 | 4.9169 | 0.8008 | 0.7171 | 0.7301 | 0.2024 |
| AP | 662106 | 34635 | 0.0523 | tabular_network@tabular | 10 | 0.1709 ± 0.0050 [0.1638-0.1795] | 3.2665 ± 0.0954 [3.1315-3.4315] | 0.7302 ± 0.0041 [0.7246-0.7366] | 0.6268 ± 0.0039 [0.6213-0.6316] | 0.6995 ± 0.0014 [0.6963-0.7007] | 0.2166 ± 0.0021 [0.2134-0.2199] |
| AP | 662106 | 34635 | 0.0523 | tabular_network@enriched | 10 | 0.1702 ± 0.0085 [0.1562-0.1856] | 3.2529 ± 0.1621 [2.9866-3.5488] | 0.7678 ± 0.0038 [0.7625-0.7739] | 0.6857 ± 0.0037 [0.6813-0.6928] | 0.7107 ± 0.0021 [0.7068-0.7137] | 0.2103 ± 0.0025 [0.2058-0.2138] |
| AP | 662106 | 34635 | 0.0523 | coords_only | 1 | 0.0535 | 1.0237 | 0.4973 | 0.5000 | 0.5000 | 0.2428 |
| AP | 662106 | 34635 | 0.0523 | cell_id_only | 1 | 0.0523 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.2348 |
| AP | 662106 | 34635 | 0.0523 | static_only | 1 | 0.0673 | 1.2869 | 0.5913 | 0.5722 | 0.5000 | 0.2377 |

### 5g. Paired ISO-week cluster bootstrap on the January-July rows: DESCRIPTIVE, POST HOC, NO VERDICTS
Differences are arm minus reference, on the saved fresh predictions, with the baseline's paired ISO-week cluster bootstrap (2,000 draws, 99% percentile interval). No verdict is given and none of these contrasts was declared; log loss uses the corrected probabilities and unit weights as in 5a.

| state | arm | reference | rows | onsets | clusters | ΔAP | ΔAP 99% lo | ΔAP 99% hi | ΔAP SE | Δlog loss | Δlog loss 99% lo | Δlog loss 99% hi | ΔBrier | ΔBrier 99% lo | ΔBrier 99% hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UK | logistic_with_burn@tabular | boosting@tabular | 508703 | 22970 | 167 | 0.0359 | 0.0162 | 0.0592 | 0.0089 | -0.0278 | -0.0370 | -0.0202 | -0.0026 | -0.0039 | -0.0015 |
| UK | boosting@enriched | fwi_season | 508703 | 22970 | 167 | -0.0133 | -0.0296 | 0.0035 | 0.0066 | -0.0073 | -0.0264 | 0.0121 | 0.0092 | 0.0055 | 0.0136 |
| UK | boosting@enriched | logistic_with_burn@enriched | 508703 | 22970 | 167 | -0.0469 | -0.0846 | -0.0160 | 0.0135 | -0.0001 | -0.0162 | 0.0151 | 0.0104 | 0.0071 | 0.0144 |
| AP | logistic_with_burn@tabular | boosting@tabular | 662106 | 34635 | 184 | -0.0890 | -0.1222 | -0.0579 | 0.0122 | 0.0054 | 0.0008 | 0.0109 | 0.0013 | 0.0007 | 0.0020 |
| AP | boosting@enriched | fwi_season | 662106 | 34635 | 184 | 0.1570 | 0.1182 | 0.1935 | 0.0145 | -0.0258 | -0.0352 | -0.0181 | -0.0031 | -0.0043 | -0.0021 |
| AP | boosting@enriched | logistic_with_burn@enriched | 662106 | 34635 | 184 | 0.0835 | 0.0513 | 0.1173 | 0.0127 | -0.0093 | -0.0147 | -0.0045 | -0.0013 | -0.0024 | -0.0006 |

### 5h. Neighbourhood-block shift for `tabular_network@enriched` (post hoc, read-only)
Mean, SD and 99th percentile of the 12 neighbourhood columns for the rows the network was standardised on ("fit": fit rows 2019-2024 without the early-stopping year 2025, after the 2,000,000-row cap), the AP rows and the UK rows that were scored. The network standardises each column with the mean and SD of the fit rows ("fit" mean and SD in the table; SD plus 1e-6).

| column | mean fit | mean AP | mean UK | sd fit | sd AP | sd UK | p99 fit | p99 AP | p99 UK |
|---|---|---|---|---|---|---|---|---|---|
| ring_8_tree_cover | 0.514 | 0.507 | 0.717 | 0.219 | 0.225 | 0.184 | 0.947 | 0.981 | 0.969 |
| ring_8_burn_count | 0.512 | 0.349 | 0.880 | 0.593 | 0.527 | 0.654 | 2.415 | 2.208 | 2.242 |
| ring_8_years_since_burn | 7.452 | 8.211 | 5.618 | 2.623 | 2.476 | 2.979 | 10.000 | 10.000 | 10.000 |
| ring_8_vegetation_index | 0.494 | 0.560 | 0.560 | 0.115 | 0.106 | 0.119 | 0.762 | 0.807 | 0.759 |
| ring_8_fire_weather_index | 30.317 | 23.388 | 10.852 | 16.618 | 14.924 | 12.878 | 69.782 | 57.487 | 56.190 |
| ring_8_onsets_recent | 3.342 | 2.535 | 2.910 | 4.425 | 3.566 | 4.046 | 18.000 | 16.000 | 18.000 |
| ring_24_tree_cover | 0.494 | 0.488 | 0.708 | 0.207 | 0.212 | 0.176 | 0.910 | 0.954 | 0.940 |
| ring_24_burn_count | 0.492 | 0.335 | 0.886 | 0.529 | 0.438 | 0.597 | 2.252 | 1.799 | 2.169 |
| ring_24_years_since_burn | 7.551 | 8.267 | 5.610 | 2.281 | 2.088 | 2.700 | 10.000 | 10.000 | 10.000 |
| ring_24_vegetation_index | 0.491 | 0.556 | 0.554 | 0.111 | 0.101 | 0.119 | 0.748 | 0.796 | 0.742 |
| ring_24_fire_weather_index | 30.335 | 23.422 | 10.836 | 16.574 | 14.866 | 12.702 | 69.730 | 57.429 | 55.732 |
| ring_24_onsets_recent | 9.768 | 7.537 | 8.610 | 11.909 | 9.482 | 11.000 | 48.000 | 43.000 | 49.000 |

Standardised values the network saw (value minus fit mean, divided by fit SD) in the scored rows:

| column | z_min AP | z_p99 AP | z_max AP | z_min UK | z_p99 UK | z_max UK |
|---|---|---|---|---|---|---|
| ring_8_tree_cover | -2.19 | 2.13 | 2.21 | -2.11 | 2.08 | 2.19 |
| ring_8_burn_count | -0.86 | 2.86 | 3.60 | -0.86 | 2.92 | 3.86 |
| ring_8_years_since_burn | -2.81 | 0.97 | 0.97 | -2.81 | 0.97 | 0.97 |
| ring_8_vegetation_index | -4.36 | 2.71 | 3.47 | -4.65 | 2.30 | 2.99 |
| ring_8_fire_weather_index | -1.82 | 1.63 | 2.91 | -1.82 | 1.56 | 3.80 |
| ring_8_onsets_recent | -0.76 | 2.86 | 5.35 | -0.76 | 3.31 | 6.48 |
| ring_24_tree_cover | -2.23 | 2.22 | 2.40 | -2.13 | 2.15 | 2.25 |
| ring_24_burn_count | -0.93 | 2.47 | 3.42 | -0.93 | 3.17 | 3.98 |
| ring_24_years_since_burn | -3.16 | 1.07 | 1.07 | -3.26 | 1.07 | 1.07 |
| ring_24_vegetation_index | -3.66 | 2.75 | 3.59 | -4.76 | 2.26 | 2.93 |
| ring_24_fire_weather_index | -1.83 | 1.63 | 2.90 | -1.83 | 1.53 | 3.76 |
| ring_24_onsets_recent | -0.82 | 2.79 | 5.73 | -0.82 | 3.29 | 5.90 |

- Largest standardised value: AP **5.73** (`ring_24_onsets_recent`), UK **6.48** (`ring_8_onsets_recent`); smallest: AP -4.36 (`ring_8_vegetation_index`), UK -4.76 (`ring_24_vegetation_index`). Largest absolute value in these columns: AP 5.73, UK 6.48; no value exceeds 10.
- Largest shift of a column mean from the fit mean, in fit SDs: AP 0.58 (`ring_24_vegetation_index`), UK 1.18 (`ring_24_fire_weather_index`).

What the network scored on rows split by their largest standardised neighbourhood value (saved predictions; thresholds 5 and 10 fixed before looking; AP of the mean over seeds' individual APs):

| state | group | rows | share_of_rows | onsets | prevalence | network_tabular_AP_mean_of_seeds | network_enriched_AP_mean_of_seeds | boosting_enriched_AP | boosting_tabular_AP |
|---|---|---|---|---|---|---|---|---|---|
| AP | max |z| <= 5 | 782671 | 1.0000 | 37668 | 0.0481 | 0.2073 | 0.1144 | 0.2944 | 0.2493 |
| AP | 5 < max |z| <= 10 | 23 | 0.0000 | 8 | 0.3478 | 0.6031 | 0.5995 | 0.6034 | 0.5639 |
| AP | max |z| > 10 | 0 | 0.0000 | 0 |  |  |  |  |  |
| AP | all rows | 782694 | 1.0000 | 37676 | 0.0481 | 0.2074 | 0.1144 | 0.2945 | 0.2494 |
| UK | max |z| <= 5 | 590746 | 0.9998 | 27466 | 0.0465 | 0.0812 | 0.1234 | 0.0978 | 0.1074 |
| UK | 5 < max |z| <= 10 | 124 | 0.0002 | 45 | 0.3629 | 0.3023 | 0.3341 | 0.4245 | 0.3367 |
| UK | max |z| > 10 | 0 | 0.0000 | 0 |  |  |  |  |  |
| UK | all rows | 590870 | 1.0000 | 27511 | 0.0466 | 0.0813 | 0.1238 | 0.0979 | 0.1075 |

AP of the same predictions by month group (every row of a month in which the state has matched non-fire days carries both labels; the other months are onset-only):

| state | months | rows | onsets | prevalence | network tabular AP (mean of seeds) | network enriched AP (mean of seeds) | boosting@enriched AP |
|---|---|---|---|---|---|---|---|
| AP | months 1-7 | 662106 | 34635 | 0.0523 | 0.1709 | 0.1702 | 0.2572 |
| AP | months 8-11 | 1666 | 1666 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| AP | month 12 | 118922 | 1375 | 0.0116 | 0.0243 | 0.0121 | 0.0296 |
| AP | all months | 782694 | 37676 | 0.0481 | 0.2074 | 0.1144 | 0.2945 |
| UK | months 1-7 | 508703 | 22970 | 0.0452 | 0.0689 | 0.1072 | 0.0814 |
| UK | months 8-11 | 1948 | 1948 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| UK | month 12 | 80219 | 2593 | 0.0323 | 0.0462 | 0.0917 | 0.1097 |
| UK | all months | 590870 | 27511 | 0.0466 | 0.0813 | 0.1238 | 0.0979 |

**Is the AP collapse explained by the neighbourhood block?** The collapse is `tabular_network@enriched` AP 0.1144 in AP, against 0.2074 for `tabular_network@tabular` and 0.2945 for `boosting@enriched` on the same rows. By these numbers extreme neighbourhood values do not explain it: 782,671 of 782,694 AP rows (0.99997 of the rows) have no standardised neighbourhood value above 5, and on them the network's AP is 0.1144, the same as on all rows; the 23 rows above 5 are not what lowers it (network AP there 0.5995). On the January-July rows the enriched network's AP in AP is 0.1702 against 0.1709 for the tabular network (5f), so the gap between 0.1144 and the tabular network arises in the other months (table above). These checks do not test the weather-window or anomaly blocks, and do not rule out an effect of the moderate shifts of the column means shown above; they only show that the collapse is not carried by a few extreme neighbourhood values. For UK the same split gives network AP 0.1234 on 590,746 rows with no value above 5 (all rows 0.1238).

All arms and the diagnostic arms are in `artifacts/results/fresh_predictions.parquet`. The sealed season (2026 January-July) is unavailable and was not scored (deviation e). Comparison with the earlier Uttarakhand reference result: `docs/decisions/2026-10-01-reference-reconciliation.md`.

## 6. Numbers that changed from the Word document (earlier reference values)
**INTERNAL ONLY, NOT FOR THE PAPER.** This table is a working comparison between the Word document's earlier reference numbers and the baseline tables; it must not be quoted or reproduced in the paper.

Old values are those the Word document states, taken from earlier reference runs, not from baseline tables. New values are read from the saved tables.

| Item | Old value | New value | Source file |
|---|---|---|---|
| Holdout 2025, trained states: boosting AP | 0.200 | 0.200 (tabular), 0.246 (enriched), fold 2025 fitted on 2019-2024 | `baseline_metrics_by_year.csv` (year 2025) |
| Holdout 2025: cell-history climatology AP | 0.117 | 0.117 | `baseline_metrics_by_year.csv` |
| Uttarakhand (UK): boosting AP | 0.059 | 0.107 (tabular), 0.098 (enriched) | `fresh_predictions.parquet` |
| Uttarakhand (UK): boosting log loss | 0.286 | 0.221 (tabular), 0.202 (enriched); climatology 0.212 | `fresh_predictions.parquet` |
| Uttarakhand (UK): FWI+season AP | 0.107 | 0.087 | `fresh_predictions.parquet` |
| True rate | 0.0147 | 0.0147 (`prior_shift.true_rate`; the fresh test uses it directly) | `configs/baseline.yaml` |
| Enriched input columns | 251 | 251 (42 tabular + 209 enriched: 180 windows, 17 anomalies, 12 neighbourhood) | `dataset.enriched_names()` |
| Fit cap | 2,000,000 | 2,000,000 (verified 2,000,000 fit rows in folds 2022-2025 and the fresh fit) | `configs/baseline.yaml` |
| Tuning fits | 600,000 | 600,000 | `configs/baseline.yaml` |
| Tuning trials | 96 trials | 24 trials × 4 folds (12 trials for boosting and 12 for the network, each scored on the validation years 2021-2024; `tuning_trials.csv` has one row per trial and fold) | `tuning_trials.csv` |
| Network seeds | 10 | 10 | `configs/models.yaml` |
| Tests passing | 161 | 169 passed | `tests/` |
| Reproduction checks | 133, 0 failures | 133 pass, 0 fail (written 2026-10-01 02:50; scope: labels, features, earlier holdout, leave-year-out 2024, out-of-fold, determinism; not rolling-origin, enriched, network or fresh scoring) | `wildfire-research/studies/reproduction/REPRODUCTION.md` |
| Peak memory | about 7 GB (was about 10 GB) | still an estimate, not a measured peak: about 7 GB for the 2M-row enriched logistic fit (float64 matrix 3.7 GB plus a 2.7 GB dataset); the old path was about 10 GB. The change is bit-identical (deviation j) | `STATUS.md` |

The first five rows were reconciled against the earlier reference runs; the result, including the differences found and the one measured effect of the row set, is in `docs/decisions/2026-10-01-reference-reconciliation.md`. In short: the earlier 0.107 is the same-day weather arm, not the next-day FWI+season arm; the UK test rows now include every season month, not January-July only (the specification declares no month rule for the fresh test, only for validation years); and the 2025 boosting AP agrees to four decimals although the predictions differ.

## 6b. Settings to quote (deviation i): `tuned.json` and `models.yaml` side by side with the specification text
| Item | `artifacts/results/tuned.json` (used by the results) | `configs/models.yaml` / `configs/baseline.yaml` | Specification text (`docs/baseline_spec.md`) |
|---|---|---|---|
| Boosting max depth | 8 | 6 (`arms.boosting.params`; used only by the older single-split pipelines) | not stated |
| Boosting learning rate | 0.05 | 0.1 | not stated |
| Boosting iterations | 500 | 300 | not stated |
| Boosting L2 / minimum leaf | 0 / 100 | not listed | not stated |
| Boosting search space | | `boosting_space`: depth [4, 6, 8], learning rate [0.05, 0.1], iterations [200, 300, 500], L2 [0, 1], leaf [20, 100] | section 6: "12 random-search trials each ... Search spaces are in `configs/baseline.yaml`" |
| Network width | 256 | search [128, 256, 512] | "MLP 3 layers of 256" |
| Network depth (hidden layers) | 2 (two hidden layers plus a one-unit output layer) | search [2, 3, 4] | "3 layers" (not defined further) |
| Network dropout | 0.1 | search [0.0, 0.1, 0.2] | 0.1 |
| Network learning rate | 0.0003 | search [0.001, 0.0003] | "AdamW" (no rate) |
| Network weight decay | 0.0001 | search [0.0001, 0.01] | "weight decay" (no value) |
| Network training | batch 4096, at most 20 epochs, stop after 2 epochs without a gain of 1e-06, 10 seeds | the same (`network:`) | "cross-entropy with the design weights, early stopping on validation log loss inside the training years, 10 seeds" |
| Logistic C | land cover plus season 1.0; the other logistic arms 0.01 | the same (used) | not stated |

**Quote `tuned.json`**: the boosting settings and the network architecture the results were produced with. State the network as two hidden layers of 256 units with dropout 0.1 and one output unit, not as "3 layers of 256" and not with the `models.yaml` boosting values (depth 6, learning rate 0.1, 300 iterations), which the baseline folds, fresh scoring, ablations and notebooks do not use.

## 7. Full deviation list
Status: written 2026-10-01 after the rolling-origin run, the sanity checks and the notebook run, before the baseline tables. Nothing here changes the stamped specification; each item says what differs from the declared plan or from the paper's natural reading, and whether it was measured.

| # | Deviation | Effect | Where recorded |
|---|---|---|---|
| a | Tuning fits use 600,000 rows; the final fits use the declared 2,000,000 | Settings were chosen on smaller fits than the ones reported. Not measured | `configs/baseline.yaml` (`row_cap`), `2026-10-01-validation-rows-and-fit-caps.md` |
| b | Tuning log loss is unweighted and aligned to the validation sample's own onset share; the final tables use design weights and per-state true rates | The selection score and the reported log loss are on different scales. Not measured | `2026-10-01-tuning-score-rule.md` |
| c | `sw1_anom` and `sw2_anom` use a same-ISO-week soil-moisture mean over 2017-2023 (weather only, no labels). Rows in 2021-2023 therefore see their own and later years; 2024 and 2025 are fully past. UK and AP rows use the same norm. The guard `no_validation_information_in_tuning` only compares 2024 < 2025 in the config and cannot see this, despite its docstring | Look-ahead in two of 42 inputs for validation years 2021-2023, which are also the tuning years. Effect unmeasured; a sensitivity run is planned | `2026-10-01-soil-anomaly-deviation.md`, `2026-10-01-soil-moisture-anomaly-look-ahead.md` |
| d | Logistic arms stop at the 300-iteration cap with sklearn's convergence warning, in every fold checked (2022 refit, 2023 profiling fits, notebook fit for 2024) | Coefficients are those of a 300-iteration stop, not the optimum; any change of rounding moves probabilities by about 1e-3. Not every arm and fold was checked | `configs/models.yaml` (`logistic.max_iter`) |
| e | Sealed season 2026 Jan-Jul declared unavailable; no sealed score exists. Partial pulls on disk (FIRMS, ERA5 2026-01 and 2026-02, NDVI) were made after the decision, are not used, and the March request is abandoned. The burn-free arms named in the earlier decision are not implemented | The sealed test is not part of the results; 2025 is not used in its place | `2026-10-01-sealed-season-unavailable.md` (with the added note), `2026-10-01-burn-free-arms-for-the-sealed-season.md` |
| f | 2025 is a development fold (scored in earlier work), not untouched | Its numbers are not a clean test | `docs/evaluation.md`, `docs/baseline_spec.md` |
| g | Fresh-state test: UK and AP only (KA, TN, KL have no usable cells), unit design weights, the configured population rate 0.0147; scored once | Log loss and Brier there are less comparable than AP and the AUC views | `2026-10-01-fresh-state-scope.md`, `2026-10-01-fresh-test-correction.md` |
| h | Network trains on the fit years before the last one and stops early on the last fit year, so it sees one year fewer than the other arms | Network and the other arms are not fitted on identical years | `2026-10-01-network-early-stopping-year.md` |
| i | Settings the results use differ from `configs/models.yaml` and from the wording of the specification (section 4). Used (`artifacts/results/tuned.json`, via `ladder_fit.settings_for` and `tuned["network"]`): boosting depth 8, learning rate 0.05, 500 iterations, minimum leaf 100, L2 0; network width 256, depth 2 (two hidden layers of 256 plus the output layer), dropout 0.1, learning rate 0.0003, weight decay 0.0001. `configs/models.yaml` `arms.boosting.params` (depth 6, learning rate 0.1, 300 iterations) is read only by the older single-split pipelines (`holdout`, `leave_year_out`, `out_of_fold`) and not by the rolling-origin folds, the fresh scoring, the ablations or the notebooks. Specification section 4 says "MLP 3 layers of 256" and gives no boosting settings; section 6 sends the search spaces to `configs/baseline.yaml` (network depth in {2, 3, 4}, width in {128, 256, 512}). Side-by-side table: `docs/PAPER_HANDOFF.md` section 6b | The paper must quote `tuned.json` (what was fitted), with the architecture stated as two hidden layers of 256 units and one output unit; it must not quote `models.yaml` boosting values or "3 layers of 256" as the fitted network. The logistic C values (0.01 for the weather and fire-weather arms, 1.0 for land cover plus season) are the ones in `models.yaml`, which are used | `artifacts/results/tuned.json`, `configs/models.yaml`, `configs/baseline.yaml`, `docs/baseline_spec.md` section 4 |
| j | Rolling-origin folds 2021 and 2022 were produced by code before commit d55a8ca (5af5654); 2023-2025 by d55a8ca | d55a8ca changed only how the logistic arm passes data to scikit-learn (float64, no float32 copies) and added per-model checkpoints. Verified bit-identical: the 2022 `logistic_with_burn@enriched` refit (max abs difference 0.0 on 839,785 rows) and the notebook refits of logistic, boosting and network seed 0 against the saved 2024 predictions (0.0 on 726,279 rows). Not re-verified: 2021, and the other 2022 arms | STATUS.md, `docs/PAPER_HANDOFF.md` |
| k | One row set for every arm (about 0.5% of rows dropped for missing values, nothing imputed) | Same rows for all arms | `2026-10-01-common-rows-and-missing-values.md` |
| l | The specification (section 9 and `configs/baseline.yaml` `guards.required`) names `identity_gate` without defining it. The implemented meaning is row disjointness: no cell-day is both a fit row and a validation row in any rolling-origin fold (2021-2025), checked on the actual fit and validation rows. No threshold. It is **not** a test against cell-identity shortcuts: that question is covered by the diagnostic arms (`coords_only`, `cell_id_only`, `static_only`), which are reported and ungated | The gate protects against fit and validation rows overlapping, nothing more; the place-shortcut is read from the diagnostic arms. The guard for `no_validation_information_in_tuning_or_climatology` is implemented as `no_validation_information_in_tuning` (see c). The power gate is implemented as a self-test of the decision rule, and the saved contrasts obey it | `src/wildfire/guards/baseline_guards.py` (`identity_gate`, `power_gate`), `docs/baseline_spec.md` section 9 |
| m | Ablations A, B1 and B2 of enriched boosting are secondary and post hoc (defined after the baseline tables) and do not change the baseline or the freeze | Their verdict labels are descriptive only | `2026-10-01-ablations.md` |
| n | Tooling change: `scripts/check_names.py` no longer scans the payload of notebook image outputs (`data` fields `image/png`, `image/jpeg`, `image/svg+xml`), because random base64 text produced label-like tokens and blocked committing executed notebooks. Every source cell, every text, stream and other output field and the notebook metadata are still scanned with the unchanged rule; only those three image fields are skipped | No change to the rule for text. A test with an executed-notebook fixture checks that label-like text in a source cell, a stream output, a text output next to an image and the metadata still fails, that an image blob passes, and that other mime types are still scanned | `scripts/check_names.py`, `tests/test_check_names.py` |
| o | Fresh-state test rows: every fire-season month of 2020-2025 through 2025-07-31 is scored (UK 590,870 rows with 27,511 onsets, AP 782,694 rows with 37,676 onsets). The specification declares no month or row rule for the fresh test: its section 2 gives only the states and "once, every access recorded in a ledger", and `configs/baseline.yaml` `fresh_geographic_test` holds the states, `scores_allowed` and the ledger name. The January-to-July rule ("Every validation year uses the rows of January to July", spec "Compute limits declared in advance", and `validation_months`) is stated for validation years. So this is **not a deviation from a declared fresh-test row set**; it is a difference from the January-July rule of the validation folds and from the earlier reference run, which scored January-July only. Scored rows of months outside a state's fire season are onset-only because matched non-fire days are drawn inside the season only (UK months 7-11: 1,967 rows; AP months 6-11: 3,080 rows). The fresh row file also holds 2019 rows (UK 77,322), which are not among the scored rows | Measured on the saved UK predictions (no re-scoring): restricting to January-July (508,703 rows, 22,970 onsets) changes `boosting@tabular` AP from 0.1075 to 0.0801 and `fwi_season` AP from 0.0869 to 0.0947. The handoff shows both row sets, the January-July table labelled a post hoc sensitivity | `docs/baseline_spec.md` sections 2 and "Compute limits", `configs/baseline.yaml`, `src/wildfire/pipelines/fresh_states.py` (`score_once`), `docs/decisions/2026-10-01-reference-reconciliation.md` |

## 8. Limits to state in the paper
- Sealed season unavailable; 2025 is a development fold; the fresh test uses two states with unit weights.
- Logistic arms stop at the 300-iteration cap; the 2021-2022 folds come from earlier code (verified identical for the 2022 enriched logistic refit and the 2024 notebook refits).
- The soil-moisture anomaly norm covers 2017-2023, so rows in 2021-2023 see their own year (deviation c); ablation A shows ΔAP +0.0010 (99% CI -0.0003 to 0.0025) when the two inputs are removed.
- No causal claim; the place shortcut is read from the diagnostic arms (reported, not gated).
