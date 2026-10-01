# Post hoc diagnostic: validation rows split by recent neighbour onsets
Status: secondary and post hoc. Requested after the baseline tables, the ablations and the fresh scoring existed. It changes nothing in the baseline, the contrasts, the freeze (tag `baseline-final`) or any verdict. Read-only: it reads the saved rolling-origin predictions and the neighbourhood array and refits nothing. Script: `docs/methodology/post_hoc_onsets_recent_strata.py`; outputs: `artifacts/results/post_hoc_onsets_recent_strata.csv` and `post_hoc_onsets_recent_gain.csv` (written after the freeze, so not in the frozen artifact manifest).

Definition. The neighbourhood block's `ring_8_onsets_recent` (primary) is the number of fire onsets in the 8 neighbouring cells over the 28 days before the target day, exactly as the model reads it. Rows with 0 form the zero stratum, rows above 0 the other. `ring_24_onsets_recent` (the 24-cell ring, same definition) is a secondary run. Arms: `boosting@tabular` and `boosting@enriched`, scores as saved. AUC is the pooled AUC inside the stratum, AP is on the raw scores. Prevalence is the sampled-row onset share (about 1 in 21 overall), not a true rate. All validation years 2021-2025, 4,521,002 rows; 2025 is a development fold.

## Pooled, primary stratifier (`ring_8_onsets_recent`)
| Stratum | Rows (share) | Onsets | Prevalence | Share of all positives | AUC tabular / enriched | AP tabular / enriched |
|---|---|---|---|---|---|---|
| zero | 1,463,672 (32.4%) | 14,211 | 0.0097 | 5.9% | 0.7787 / 0.7883 | 0.0935 / 0.0969 |
| above zero | 3,057,330 (67.6%) | 226,284 | 0.0740 | 94.1% | 0.7322 / 0.7692 | 0.2023 / 0.2317 |

The enriched gain in AP, with the same paired ISO-week cluster bootstrap as the contrasts (2,000 draws):
| Stratum | ΔAP (enriched minus tabular) | 99% CI | ΔAUC (point) |
|---|---|---|---|
| zero | +0.0034 | -0.0057 to +0.0170 | +0.0095 |
| above zero | +0.0294 | +0.0202 to +0.0372 | +0.0370 |
(All rows: ΔAP +0.0352, 99% CI 0.0251 to 0.0441, the baseline contrast.)

## Per year (primary stratifier)
| Year | Stratum | Rows | Onsets | Prevalence | Share of positives | AUC tab / enr | AP tab / enr |
|---|---|---|---|---|---|---|---|
| 2021 | zero | 300,826 | 2,501 | 0.0083 | 4.4% | 0.7583 / 0.7623 | 0.0891 / 0.0644 |
| 2021 | above zero | 733,926 | 54,579 | 0.0744 | 95.6% | 0.7182 / 0.7749 | 0.1780 / 0.2118 |
| 2022 | zero | 325,005 | 3,074 | 0.0095 | 7.0% | 0.8270 / 0.8292 | 0.1203 / 0.1161 |
| 2022 | above zero | 514,780 | 40,984 | 0.0796 | 93.0% | 0.7713 / 0.7870 | 0.2627 / 0.2779 |
| 2023 | zero | 269,239 | 3,092 | 0.0115 | 6.1% | 0.7573 / 0.7652 | 0.1000 / 0.1107 |
| 2023 | above zero | 670,393 | 47,328 | 0.0706 | 93.9% | 0.7482 / 0.7698 | 0.2229 / 0.2431 |
| 2024 | zero | 263,581 | 2,888 | 0.0110 | 7.7% | 0.7924 / 0.8075 | 0.0851 / 0.1028 |
| 2024 | above zero | 462,698 | 34,791 | 0.0752 | 92.3% | 0.7550 / 0.7873 | 0.2177 / 0.2564 |
| 2025 | zero | 305,021 | 2,656 | 0.0087 | 5.2% | 0.7995 / 0.8178 | 0.0990 / 0.1130 |
| 2025 | above zero | 675,533 | 48,602 | 0.0719 | 94.8% | 0.7687 / 0.8035 | 0.2079 / 0.2512 |

## Secondary stratifier (`ring_24_onsets_recent`, pooled)
Zero stratum: 755,811 rows (16.7%), 3,716 onsets (1.6% of positives), prevalence 0.0049, AUC 0.7777 / 0.7642, AP 0.0858 / 0.0831, ΔAP -0.0027 (99% CI -0.0175 to +0.0157). Above zero: 3,765,191 rows, 236,779 onsets, prevalence 0.0629, AUC 0.7522 / 0.7921, AP 0.1946 / 0.2276, ΔAP +0.0330 (99% CI +0.0235 to +0.0412).

## Reading
- **Does the enriched gain exist in the zero stratum?** It is not demonstrated. The pooled point estimate is +0.0034 with a 99% interval of -0.0057 to +0.0170, which contains zero and also contains the 0.010 threshold, so a small gain cannot be excluded and none is shown. The per-year ΔAP in that stratum goes both ways (2021 -0.0247, 2022 -0.0042, 2023 +0.0107, 2024 +0.0177, 2025 +0.0140); ΔAUC is positive in every year but small (pooled +0.0095). With the 24-cell ring as the stratifier the zero-stratum ΔAP is -0.0027 (CI -0.0175 to +0.0157).
- **Where the gain is.** The enriched AP gain sits in rows with recent neighbour onsets: +0.0294 (CI +0.0202 to +0.0372) in the above-zero stratum, which holds 94.1% of all positives and 67.6% of rows. This agrees with the neighbourhood-only ablation carrying the enriched gain.
- **Limits.** The zero stratum has few positives (14,211 pooled, 2,500 to 3,100 per year), so its interval is wide. AP is not comparable between strata because it depends on prevalence (0.0097 against 0.0740); AP divided by prevalence is 10.0 against 3.1 for enriched. `onsets_recent` is built from observed fire onsets of neighbouring cells (not from any model output); the split shows where the gain occurs and not why. No causal reading is made.
