# Paper handoff: baseline results

Generated 2026-10-01 21:31 IST from the saved tables by `docs/methodology/make_paper_handoff.py`. Numbers below are read from files under `artifacts/results/`; the paper and the Word document were not touched.

## 0. Provenance
- Frozen baseline: local tag `baseline-final` on commit `5d7d83f` (not pushed); records in `artifacts/baseline_final/` (code hash, config snapshot, data manifest, artifact manifest).
- Code that produced each result (the `code_commit` stamped in the files): baseline tables `d55a8ca`, ablation tables `0f3ac9e`, sanity checks `d55a8ca`, fresh scoring `5d7d83f`. Rolling-origin folds 2023-2025 were fitted by `d55a8ca`, folds 2021 and 2022 by the earlier code `5af5654` (deviation j). Specification commit `99c867c`.
- Guard record (`guards.json`, code `9da492c`), every guard true: timestamp_before_day, shifted_feature_fails, identity_gate, label_shuffle, coordinate_unit_test, no_validation_information_in_tuning, power_gate, geo_ledger, provenance; label-shuffle mean AUC 0.5012.
- Data and artifact records: `wildfire verify-data` (21:2x on 2026-10-01) compared `data/MANIFEST.json` with the data folders and found no differences (exit code 0, no output). The frozen artifact manifest (`artifacts/baseline_final/artifact_manifest.json`) compared with the folders now: labels and features unchanged; results changed from 39 to 44 files, the five files added after the freeze (the fresh predictions, the fresh fit prevalences, the ledger and the two strata CSVs). The data manifest record itself was copied from the committed manifest.
- Tests at this commit: 169 passed in 12.18s. Sanity checks (`sanity.csv`): global label shuffle mean AP 0.0502 against prevalence 0.0522, cell x week shuffle max AP 0.0933 against climatology 0.1167, permuted holdout labels 0.0522: all passed.
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
Every arm fitted on all years before 2026 (2,000,000-row cap) and scored on the fresh rows: AP 782,694 rows (37,676 onsets, sampled prevalence 0.0481), UK 590,870 rows (27,511 onsets, sampled prevalence 0.0466), years 2020-2025, every month of the fire season through 2025-07-31 (not restricted to January-July: deviation o). Unit design weights and the configured population rate 0.0147 are used, so log loss is less comparable than AP and the AUC views. Lift is AP divided by the state's sampled prevalence. No state-level contrast tests were declared for the fresh states; no verdicts are given.

Networks are shown as mean ± SD over the 10 seeds with the per-seed range [min-max]; every other arm has one fit. The figures AP 0.1144 (AP) and 0.1238 (UK), quoted in earlier discussion, are the AP of **`tabular_network@enriched`** (mean over 10 seeds); the tabular-level network has AP 0.2074 (AP) and 0.0813 (UK).

| state | arm | seeds | AP | lift (AP / prevalence) | AUC pooled | AUC spatial | AUC temporal | log loss |
|---|---|---|---|---|---|---|---|---|
| UK | climatology_cell | 1 | 0.0564 | 1.2110 | 0.5642 | 0.5917 | 0.5194 | 0.2120 |
| UK | landcover_season | 1 | 0.0540 | 1.1594 | 0.5538 | 0.4540 | 0.5796 | 0.2133 |
| UK | fwi_season | 1 | 0.0869 | 1.8672 | 0.6197 | 0.5445 | 0.6196 | 0.2236 |
| UK | logistic_weather_static@tabular | 1 | 0.1008 | 2.1642 | 0.6734 | 0.5803 | 0.6587 | 0.2187 |
| UK | logistic_weather_static@enriched | 1 | 0.1267 | 2.7221 | 0.7110 | 0.6583 | 0.6797 | 0.2134 |
| UK | logistic_with_burn@tabular | 1 | 0.1077 | 2.3134 | 0.6805 | 0.5932 | 0.6568 | 0.2032 |
| UK | logistic_with_burn@enriched | 1 | 0.1265 | 2.7170 | 0.7110 | 0.6592 | 0.6789 | 0.2145 |
| UK | boosting@tabular | 1 | 0.1075 | 2.3083 | 0.6642 | 0.6316 | 0.6477 | 0.2208 |
| UK | boosting@enriched | 1 | 0.0979 | 2.1032 | 0.7185 | 0.6562 | 0.7211 | 0.2025 |
| UK | tabular_network@tabular | 10 | 0.0813 ± 0.0038 [0.0738-0.0874] | 1.7469 ± 0.0823 [1.5858-1.8773] | 0.6376 ± 0.0077 [0.6270-0.6544] | 0.6141 ± 0.0040 [0.6082-0.6206] | 0.6161 ± 0.0091 [0.6057-0.6356] | 0.2861 ± 0.0128 [0.2717-0.3175] |
| UK | tabular_network@enriched | 10 | 0.1238 ± 0.0078 [0.1036-0.1313] | 2.6579 ± 0.1669 [2.2258-2.8201] | 0.7308 ± 0.0160 [0.6972-0.7508] | 0.6912 ± 0.0075 [0.6745-0.7012] | 0.7066 ± 0.0260 [0.6534-0.7472] | 0.2313 ± 0.0077 [0.2166-0.2415] |
| UK | coords_only | 1 | 0.0458 | 0.9840 | 0.4972 | 0.5160 | 0.5000 | 0.2362 |
| UK | cell_id_only | 1 | 0.0466 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.2106 |
| UK | static_only | 1 | 0.0509 | 1.0926 | 0.5404 | 0.5128 | 0.5000 | 0.2073 |
| AP | climatology_cell | 1 | 0.0887 | 1.8423 | 0.6417 | 0.6139 | 0.5693 | 0.2083 |
| AP | landcover_season | 1 | 0.0772 | 1.6039 | 0.6503 | 0.5276 | 0.6119 | 0.2154 |
| AP | fwi_season | 1 | 0.0961 | 1.9970 | 0.6674 | 0.5549 | 0.6453 | 0.2156 |
| AP | logistic_weather_static@tabular | 1 | 0.1050 | 2.1815 | 0.7055 | 0.5701 | 0.6747 | 0.2051 |
| AP | logistic_weather_static@enriched | 1 | 0.1456 | 3.0246 | 0.7565 | 0.6521 | 0.7206 | 0.1972 |
| AP | logistic_with_burn@tabular | 1 | 0.1061 | 2.2037 | 0.7071 | 0.5879 | 0.6738 | 0.2064 |
| AP | logistic_with_burn@enriched | 1 | 0.1453 | 3.0187 | 0.7567 | 0.6527 | 0.7202 | 0.1972 |
| AP | boosting@tabular | 1 | 0.2494 | 5.1808 | 0.7742 | 0.6558 | 0.7355 | 0.1921 |
| AP | boosting@enriched | 1 | 0.2945 | 6.1186 | 0.8147 | 0.7188 | 0.7600 | 0.1815 |
| AP | tabular_network@tabular | 10 | 0.2074 ± 0.0092 [0.1925-0.2195] | 4.3084 ± 0.1916 [3.9993-4.5601] | 0.7527 ± 0.0055 [0.7445-0.7606] | 0.6403 ± 0.0042 [0.6338-0.6457] | 0.7319 ± 0.0021 [0.7285-0.7357] | 0.1946 ± 0.0019 [0.1918-0.1976] |
| AP | tabular_network@enriched | 10 | 0.1144 ± 0.0071 [0.1061-0.1304] | 2.3765 ± 0.1478 [2.2034-2.7094] | 0.7507 ± 0.0069 [0.7427-0.7643] | 0.6675 ± 0.0042 [0.6631-0.6757] | 0.7101 ± 0.0055 [0.7017-0.7219] | 0.2558 ± 0.0182 [0.2325-0.2842] |
| AP | coords_only | 1 | 0.0496 | 1.0309 | 0.5021 | 0.5120 | 0.5000 | 0.2242 |
| AP | cell_id_only | 1 | 0.0481 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.2172 |
| AP | static_only | 1 | 0.0602 | 1.2515 | 0.5802 | 0.5548 | 0.5000 | 0.2204 |

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

The first five rows were reconciled against the earlier reference runs; the result, including the differences found and the one measured effect of the row set, is in `docs/decisions/2026-10-01-reference-reconciliation.md`. In short: the earlier 0.107 is the same-day weather arm, not the next-day FWI+season arm; the UK test rows now include every season month, not January-July only; and the 2025 boosting AP agrees to four decimals although the predictions differ.

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
| i | Tuned settings in use (`tuned.json`): boosting depth 8, learning rate 0.05, 500 iterations, leaf 100; network width 256, depth 2. `configs/models.yaml` still lists boosting depth 6, learning rate 0.1, 300 iterations, and the specification text says a 3-layer MLP of 256 | The run uses `tuned.json` (via `ladder_fit.settings_for`); the config and spec text are stale or differ. To be checked against the spec before the paper quotes them | `configs/models.yaml`, `artifacts/results/tuned.json`, `docs/baseline_spec.md` section 4 |
| j | Rolling-origin folds 2021 and 2022 were produced by code before commit d55a8ca (5af5654); 2023-2025 by d55a8ca | d55a8ca changed only how the logistic arm passes data to scikit-learn (float64, no float32 copies) and added per-model checkpoints. Verified bit-identical: the 2022 `logistic_with_burn@enriched` refit (max abs difference 0.0 on 839,785 rows) and the notebook refits of logistic, boosting and network seed 0 against the saved 2024 predictions (0.0 on 726,279 rows). Not re-verified: 2021, and the other 2022 arms | STATUS.md, `docs/PAPER_HANDOFF.md` |
| k | One row set for every arm (about 0.5% of rows dropped for missing values, nothing imputed) | Same rows for all arms | `2026-10-01-common-rows-and-missing-values.md` |
| l | The specification (section 9 and `configs/baseline.yaml` `guards.required`) names `identity_gate` without defining it. The implemented meaning is row disjointness: no cell-day is both a fit row and a validation row in any rolling-origin fold (2021-2025), checked on the actual fit and validation rows. No threshold. It is **not** a test against cell-identity shortcuts: that question is covered by the diagnostic arms (`coords_only`, `cell_id_only`, `static_only`), which are reported and ungated | The gate protects against fit and validation rows overlapping, nothing more; the place-shortcut is read from the diagnostic arms. The guard for `no_validation_information_in_tuning_or_climatology` is implemented as `no_validation_information_in_tuning` (see c). The power gate is implemented as a self-test of the decision rule, and the saved contrasts obey it | `src/wildfire/guards/baseline_guards.py` (`identity_gate`, `power_gate`), `docs/baseline_spec.md` section 9 |
| m | Ablations A, B1 and B2 of enriched boosting are secondary and post hoc (defined after the baseline tables) and do not change the baseline or the freeze | Their verdict labels are descriptive only | `2026-10-01-ablations.md` |
| n | Tooling change: `scripts/check_names.py` no longer scans the payload of notebook image outputs (`data` fields `image/png`, `image/jpeg`, `image/svg+xml`), because random base64 text produced label-like tokens and blocked committing executed notebooks. Every source cell, every text, stream and other output field and the notebook metadata are still scanned with the unchanged rule; only those three image fields are skipped | No change to the rule for text. A test with an executed-notebook fixture checks that label-like text in a source cell, a stream output, a text output next to an image and the metadata still fails, that an image blob passes, and that other mime types are still scanned | `scripts/check_names.py`, `tests/test_check_names.py` |
| o | Fresh-state test rows: every fire-season month of 2020-2025 through 2025-07-31 is scored (UK 590,870 rows with 27,511 onsets, AP 782,694 rows with 37,676 onsets), not January-July only as in the validation folds. The fresh row file also holds 2019 rows (UK 77,322), which are not among the scored rows. The specification declares no row rule for the fresh test | Measured on the saved UK predictions: restricting to January-July (508,703 rows, 22,970 onsets) changes `boosting@tabular` AP from 0.1075 to 0.0801 and `fwi_season` AP from 0.0869 to 0.0947. The fresh table in the handoff uses all scored rows. Nothing was re-scored | `src/wildfire/pipelines/fresh_states.py` (`score_once`), `docs/decisions/2026-10-01-reference-reconciliation.md` |

## 8. Limits to state in the paper
- Sealed season unavailable; 2025 is a development fold; the fresh test uses two states with unit weights.
- Logistic arms stop at the 300-iteration cap; the 2021-2022 folds come from earlier code (verified identical for the 2022 enriched logistic refit and the 2024 notebook refits).
- The soil-moisture anomaly norm covers 2017-2023, so rows in 2021-2023 see their own year (deviation c); ablation A shows ΔAP +0.0010 (99% CI -0.0003 to 0.0025) when the two inputs are removed.
- No causal claim; the place shortcut is read from the diagnostic arms (reported, not gated).
