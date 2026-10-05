# Results table: frozen baseline against the deep multimodal model

Folds 2024 and 2023: validation years the models were not trained on. Both models are scored on the same rows with the same labels and weights. Probabilities are corrected from the matched-negative sample back to the true fire rate and are not calibrated further. The top-1% and top-5% rows are chosen among the cells of each day. "Deep minus baseline" is positive where the deep model is higher, which is better for every metric except Brier score, log loss, calibration error and false alarm ratio. **Only average precision has a confidence interval** (exact paired bootstrap over week blocks); every other number is a point estimate.

## Fold 2024

Average precision with 95% interval: baseline 0.2267 (0.1704 to 0.2679); deep model 0.2913 (0.2336 to 0.3350); difference +0.0646 (+0.0543 to +0.0734).

| Metric | Frozen baseline | Deep multimodal model | Deep minus baseline |
|---|---|---|---|
| Average precision (pooled) | 0.2267 | 0.2913 | +0.0646 |
| Average precision, ranking cells within each day | 0.1932 | 0.2432 | +0.0500 |
| Average precision, ranking days within each cell | 0.3927 | 0.4246 | +0.0319 |
| AUROC (pooled) | 0.8346 | 0.8575 | +0.0229 |
| AUROC, ranking cells within each day | 0.8081 | 0.8335 | +0.0254 |
| AUROC, ranking days within each cell | 0.7957 | 0.8117 | +0.0160 |
| Precision, top 1% of cells per day | 0.1033 | 0.1574 | +0.0541 |
| Recall, top 1% of cells per day | 0.0567 | 0.0738 | +0.0171 |
| Lift, top 1% of cells per day | 6.96 | 9.78 | +2.82 |
| Precision, top 5% of cells per day | 0.0684 | 0.0888 | +0.0204 |
| Recall, top 5% of cells per day | 0.2092 | 0.2528 | +0.0436 |
| Lift, top 5% of cells per day | 4.96 | 6.34 | +1.38 |
| Precision, top 10% of cells per day | 0.0546 | 0.0655 | +0.0109 |
| Brier score (lower is better) | 0.01360 | 0.01333 | -0.00027 |
| Brier score of the cell climatology (reference) | 0.01397 | 0.01397 | +0.00000 |
| Brier skill score against the cell climatology | 0.0266 | 0.0458 | +0.0192 |
| Log loss (lower is better) | 0.0628 | 0.0600 | -0.0027 |
| Calibration error, ECE (lower is better) | 0.00048 | 0.00022 | -0.00026 |
| Alert threshold (probability, fixed on the other fold) | 0.0656 | 0.0778 | +0.0121 |
| Share of cell-days alerted | 0.0418 | 0.0350 | -0.0068 |
| Probability of detection | 0.2861 | 0.3173 | +0.0312 |
| False alarm ratio (lower is better) | 0.9019 | 0.8701 | -0.0318 |
| Critical success index | 0.0788 | 0.1015 | +0.0227 |
| Rows | 725,441 | 725,441 |  |
| Fire onsets | 36,841 | 36,841 |  |
| Fire rate (population-weighted) | 0.0143 | 0.0143 | +0.0000 |

## Fold 2023

Average precision with 95% interval: baseline 0.1797 (0.1419 to 0.2192); deep model 0.2532 (0.2108 to 0.2872); difference +0.0734 (+0.0549 to +0.0872).

| Metric | Frozen baseline | Deep multimodal model | Deep minus baseline |
|---|---|---|---|
| Average precision (pooled) | 0.1797 | 0.2532 | +0.0734 |
| Average precision, ranking cells within each day | 0.1775 | 0.2191 | +0.0416 |
| Average precision, ranking days within each cell | 0.3201 | 0.3762 | +0.0561 |
| AUROC (pooled) | 0.8059 | 0.8394 | +0.0335 |
| AUROC, ranking cells within each day | 0.7795 | 0.8054 | +0.0258 |
| AUROC, ranking days within each cell | 0.7749 | 0.8017 | +0.0268 |
| Precision, top 1% of cells per day | 0.1092 | 0.1568 | +0.0476 |
| Recall, top 1% of cells per day | 0.0498 | 0.0676 | +0.0178 |
| Lift, top 1% of cells per day | 5.69 | 8.30 | +2.60 |
| Precision, top 5% of cells per day | 0.0711 | 0.0901 | +0.0190 |
| Recall, top 5% of cells per day | 0.1871 | 0.2290 | +0.0418 |
| Lift, top 5% of cells per day | 4.14 | 5.24 | +1.09 |
| Precision, top 10% of cells per day | 0.0580 | 0.0702 | +0.0122 |
| Brier score (lower is better) | 0.01712 | 0.01673 | -0.00039 |
| Brier score of the cell climatology (reference) | 0.01754 | 0.01754 | +0.00000 |
| Brier skill score against the cell climatology | 0.0237 | 0.0460 | +0.0223 |
| Log loss (lower is better) | 0.0784 | 0.0744 | -0.0040 |
| Calibration error, ECE (lower is better) | 0.00079 | 0.00155 | +0.00075 |
| Alert threshold (probability, fixed on the other fold) | 0.0755 | 0.0929 | +0.0174 |
| Share of cell-days alerted | 0.0433 | 0.0267 | -0.0166 |
| Probability of detection | 0.2296 | 0.2286 | -0.0010 |
| False alarm ratio (lower is better) | 0.9048 | 0.8461 | -0.0587 |
| Critical success index | 0.0722 | 0.1013 | +0.0291 |
| Rows | 936,380 | 936,380 |  |
| Fire onsets | 47,168 | 47,168 |  |
| Fire rate (population-weighted) | 0.0180 | 0.0180 | +0.0000 |

## Quiet stratum, fold 2024 (no neighbouring fire detection in the previous 7 days)

Average precision with 95% interval: baseline 0.0343 (0.0287 to 0.0409); deep model 0.0640 (0.0495 to 0.0831); difference +0.0297 (+0.0180 to +0.0452). The alert threshold is the one fixed for the whole fold.

| Metric | Frozen baseline | Deep multimodal model | Deep minus baseline |
|---|---|---|---|
| Average precision (pooled) | 0.0343 | 0.0640 | +0.0297 |
| Average precision, ranking cells within each day | 0.0790 | 0.1189 | +0.0398 |
| Average precision, ranking days within each cell | 0.4994 | 0.4941 | -0.0052 |
| AUROC (pooled) | 0.7885 | 0.8078 | +0.0193 |
| AUROC, ranking cells within each day | 0.7363 | 0.7650 | +0.0287 |
| AUROC, ranking days within each cell | 0.7270 | 0.7124 | -0.0146 |
| Precision, top 1% of cells per day | 0.0184 | 0.0269 | +0.0085 |
| Recall, top 1% of cells per day | 0.0797 | 0.1023 | +0.0227 |
| Lift, top 1% of cells per day | 8.15 | 10.74 | +2.59 |
| Precision, top 5% of cells per day | 0.0087 | 0.0110 | +0.0024 |
| Recall, top 5% of cells per day | 0.2171 | 0.2610 | +0.0439 |
| Lift, top 5% of cells per day | 4.65 | 5.62 | +0.97 |
| Precision, top 10% of cells per day | 0.0072 | 0.0078 | +0.0006 |
| Brier score (lower is better) | 0.00168 | 0.00166 | -0.00002 |
| Brier score of the cell climatology (reference) | 0.00181 | 0.00181 | +0.00000 |
| Brier skill score against the cell climatology | 0.0713 | 0.0834 | +0.0121 |
| Log loss (lower is better) | 0.0117 | 0.0111 | -0.0006 |
| Calibration error, ECE (lower is better) | 0.00138 | 0.00028 | -0.00111 |
| Alert threshold (probability, fixed on the other fold) | 0.0656 | 0.0778 | +0.0121 |
| Share of cell-days alerted | 0.0011 | 0.0001 | -0.0010 |
| Probability of detection | 0.0197 | 0.0197 | +0.0000 |
| False alarm ratio (lower is better) | 0.9697 | 0.7028 | -0.2669 |
| Critical success index | 0.0121 | 0.0189 | +0.0068 |
| Rows | 201,099 | 201,099 |  |
| Fire onsets | 1,368 | 1,368 |  |
| Fire rate (population-weighted) | 0.0017 | 0.0017 | +0.0000 |

## Quiet stratum, fold 2023 (no neighbouring fire detection in the previous 7 days)

Average precision with 95% interval: baseline 0.0197 (0.0172 to 0.0253); deep model 0.0347 (0.0271 to 0.0466); difference +0.0150 (+0.0084 to +0.0242). The alert threshold is the one fixed for the whole fold.

| Metric | Frozen baseline | Deep multimodal model | Deep minus baseline |
|---|---|---|---|
| Average precision (pooled) | 0.0197 | 0.0347 | +0.0150 |
| Average precision, ranking cells within each day | 0.0533 | 0.0787 | +0.0254 |
| Average precision, ranking days within each cell | 0.4331 | 0.4059 | -0.0272 |
| AUROC (pooled) | 0.7259 | 0.7603 | +0.0344 |
| AUROC, ranking cells within each day | 0.6840 | 0.7113 | +0.0273 |
| AUROC, ranking days within each cell | 0.7015 | 0.6874 | -0.0141 |
| Precision, top 1% of cells per day | 0.0124 | 0.0235 | +0.0111 |
| Recall, top 1% of cells per day | 0.0548 | 0.0876 | +0.0328 |
| Lift, top 1% of cells per day | 5.45 | 8.55 | +3.09 |
| Precision, top 5% of cells per day | 0.0087 | 0.0104 | +0.0017 |
| Recall, top 5% of cells per day | 0.1838 | 0.2086 | +0.0247 |
| Lift, top 5% of cells per day | 3.79 | 4.19 | +0.40 |
| Precision, top 10% of cells per day | 0.0073 | 0.0081 | +0.0008 |
| Brier score (lower is better) | 0.00244 | 0.00236 | -0.00007 |
| Brier score of the cell climatology (reference) | 0.00249 | 0.00249 | +0.00000 |
| Brier skill score against the cell climatology | 0.0219 | 0.0513 | +0.0294 |
| Log loss (lower is better) | 0.0172 | 0.0156 | -0.0016 |
| Calibration error, ECE (lower is better) | 0.00270 | 0.00042 | -0.00228 |
| Alert threshold (probability, fixed on the other fold) | 0.0755 | 0.0929 | +0.0174 |
| Share of cell-days alerted | 0.0027 | 0.0001 | -0.0026 |
| Probability of detection | 0.0241 | 0.0074 | -0.0167 |
| False alarm ratio (lower is better) | 0.9789 | 0.7295 | -0.2493 |
| Critical success index | 0.0114 | 0.0072 | -0.0042 |
| Rows | 208,347 | 208,347 |  |
| Fire onsets | 1,496 | 1,496 |  |
| Fire rate (population-weighted) | 0.0024 | 0.0024 | +0.0000 |

## Per state, fold 2024

Each state is ranked on its own rows.

| State | Metric | Frozen baseline | Deep multimodal model | Deep minus baseline |
|---|---|---|---|---|
| Chhattisgarh | Average precision | 0.2202 | 0.3116 | +0.0914 |
| Chhattisgarh | AUROC | 0.8447 | 0.8726 | +0.0279 |
| Chhattisgarh | Precision, top 5% per day | 0.0663 | 0.0959 | +0.0296 |
| Chhattisgarh | Recall, top 5% per day | 0.1822 | 0.2386 | +0.0564 |
| Chhattisgarh | Lift, top 5% per day | 4.01 | 5.51 | +1.50 |
| Chhattisgarh | Brier score | 0.01388 | 0.01349 | -0.00040 |
| Chhattisgarh | Brier skill score | 0.0285 | 0.0561 | +0.0277 |
| Jharkhand | Average precision | 0.3258 | 0.4291 | +0.1033 |
| Jharkhand | AUROC | 0.9081 | 0.9281 | +0.0201 |
| Jharkhand | Precision, top 5% per day | 0.0543 | 0.0714 | +0.0172 |
| Jharkhand | Recall, top 5% per day | 0.2025 | 0.2389 | +0.0363 |
| Jharkhand | Lift, top 5% per day | 4.54 | 5.55 | +1.01 |
| Jharkhand | Brier score | 0.00865 | 0.00836 | -0.00030 |
| Jharkhand | Brier skill score | 0.0341 | 0.0673 | +0.0332 |
| Maharashtra | Average precision | 0.2004 | 0.2271 | +0.0268 |
| Maharashtra | AUROC | 0.8009 | 0.8140 | +0.0132 |
| Maharashtra | Precision, top 5% per day | 0.0834 | 0.0963 | +0.0129 |
| Maharashtra | Recall, top 5% per day | 0.1867 | 0.2079 | +0.0212 |
| Maharashtra | Lift, top 5% per day | 4.10 | 4.64 | +0.54 |
| Maharashtra | Brier score | 0.01779 | 0.01763 | -0.00016 |
| Maharashtra | Brier skill score | 0.0207 | 0.0297 | +0.0089 |
| Madhya Pradesh | Average precision | 0.2417 | 0.2933 | +0.0517 |
| Madhya Pradesh | AUROC | 0.8408 | 0.8612 | +0.0204 |
| Madhya Pradesh | Precision, top 5% per day | 0.0552 | 0.0664 | +0.0113 |
| Madhya Pradesh | Recall, top 5% per day | 0.2050 | 0.2352 | +0.0302 |
| Madhya Pradesh | Lift, top 5% per day | 4.59 | 5.41 | +0.82 |
| Madhya Pradesh | Brier score | 0.00978 | 0.00964 | -0.00015 |
| Madhya Pradesh | Brier skill score | 0.0220 | 0.0366 | +0.0145 |
| Odisha | Average precision | 0.2442 | 0.3131 | +0.0689 |
| Odisha | AUROC | 0.8411 | 0.8665 | +0.0254 |
| Odisha | Precision, top 5% per day | 0.0735 | 0.0913 | +0.0178 |
| Odisha | Recall, top 5% per day | 0.2052 | 0.2389 | +0.0337 |
| Odisha | Lift, top 5% per day | 4.63 | 5.56 | +0.93 |
| Odisha | Brier score | 0.01333 | 0.01305 | -0.00029 |
| Odisha | Brier skill score | 0.0307 | 0.0515 | +0.0208 |
| Telangana | Average precision | 0.1787 | 0.2272 | +0.0485 |
| Telangana | AUROC | 0.7990 | 0.8240 | +0.0250 |
| Telangana | Precision, top 5% per day | 0.0909 | 0.1131 | +0.0222 |
| Telangana | Recall, top 5% per day | 0.1554 | 0.1889 | +0.0336 |
| Telangana | Lift, top 5% per day | 3.23 | 4.01 | +0.77 |
| Telangana | Brier score | 0.02265 | 0.02221 | -0.00044 |
| Telangana | Brier skill score | 0.0254 | 0.0445 | +0.0191 |

## Per state, fold 2023

Each state is ranked on its own rows.

| State | Metric | Frozen baseline | Deep multimodal model | Deep minus baseline |
|---|---|---|---|---|
| Chhattisgarh | Average precision | 0.1690 | 0.2608 | +0.0919 |
| Chhattisgarh | AUROC | 0.7979 | 0.8480 | +0.0501 |
| Chhattisgarh | Precision, top 5% per day | 0.0840 | 0.1108 | +0.0268 |
| Chhattisgarh | Recall, top 5% per day | 0.1823 | 0.2259 | +0.0436 |
| Chhattisgarh | Lift, top 5% per day | 3.96 | 5.06 | +1.10 |
| Chhattisgarh | Brier score | 0.01836 | 0.01780 | -0.00056 |
| Chhattisgarh | Brier skill score | 0.0272 | 0.0571 | +0.0298 |
| Jharkhand | Average precision | 0.2259 | 0.3456 | +0.1197 |
| Jharkhand | AUROC | 0.8500 | 0.8859 | +0.0359 |
| Jharkhand | Precision, top 5% per day | 0.0705 | 0.0934 | +0.0229 |
| Jharkhand | Recall, top 5% per day | 0.1949 | 0.2365 | +0.0416 |
| Jharkhand | Lift, top 5% per day | 4.32 | 5.44 | +1.12 |
| Jharkhand | Brier score | 0.01337 | 0.01291 | -0.00045 |
| Jharkhand | Brier skill score | 0.0269 | 0.0600 | +0.0330 |
| Maharashtra | Average precision | 0.1474 | 0.1894 | +0.0420 |
| Maharashtra | AUROC | 0.7765 | 0.7952 | +0.0187 |
| Maharashtra | Precision, top 5% per day | 0.0728 | 0.0908 | +0.0180 |
| Maharashtra | Recall, top 5% per day | 0.1687 | 0.2021 | +0.0334 |
| Maharashtra | Lift, top 5% per day | 3.63 | 4.45 | +0.82 |
| Maharashtra | Brier score | 0.01831 | 0.01808 | -0.00023 |
| Maharashtra | Brier skill score | 0.0121 | 0.0247 | +0.0126 |
| Madhya Pradesh | Average precision | 0.1689 | 0.2303 | +0.0613 |
| Madhya Pradesh | AUROC | 0.8004 | 0.8292 | +0.0288 |
| Madhya Pradesh | Precision, top 5% per day | 0.0532 | 0.0730 | +0.0198 |
| Madhya Pradesh | Recall, top 5% per day | 0.1689 | 0.2145 | +0.0457 |
| Madhya Pradesh | Lift, top 5% per day | 3.65 | 4.81 | +1.16 |
| Madhya Pradesh | Brier score | 0.01307 | 0.01282 | -0.00025 |
| Madhya Pradesh | Brier skill score | 0.0095 | 0.0283 | +0.0188 |
| Odisha | Average precision | 0.1959 | 0.2688 | +0.0729 |
| Odisha | AUROC | 0.8192 | 0.8508 | +0.0316 |
| Odisha | Precision, top 5% per day | 0.0909 | 0.1127 | +0.0217 |
| Odisha | Recall, top 5% per day | 0.1818 | 0.2168 | +0.0350 |
| Odisha | Lift, top 5% per day | 3.96 | 4.84 | +0.88 |
| Odisha | Brier score | 0.01949 | 0.01903 | -0.00046 |
| Odisha | Brier skill score | 0.0322 | 0.0552 | +0.0230 |
| Telangana | Average precision | 0.1793 | 0.2159 | +0.0365 |
| Telangana | AUROC | 0.7872 | 0.8216 | +0.0344 |
| Telangana | Precision, top 5% per day | 0.0735 | 0.0960 | +0.0225 |
| Telangana | Recall, top 5% per day | 0.1448 | 0.1845 | +0.0397 |
| Telangana | Lift, top 5% per day | 2.99 | 3.90 | +0.91 |
| Telangana | Brier score | 0.01989 | 0.01953 | -0.00036 |
| Telangana | Brier skill score | 0.0294 | 0.0468 | +0.0174 |

## Calibration curves (all rows)

Ten equal-count groups of predicted probability; mean predicted probability against the observed fire rate.

Fold 2024

| Group | Baseline: predicted | observed | Deep model: predicted | observed |
|---|---|---|---|---|
| 1 | 0.00044 | 0.00021 | 0.00026 | 0.00019 |
| 2 | 0.00112 | 0.00088 | 0.00070 | 0.00060 |
| 3 | 0.00215 | 0.00186 | 0.00138 | 0.00146 |
| 4 | 0.00360 | 0.00323 | 0.00244 | 0.00250 |
| 5 | 0.00560 | 0.00500 | 0.00412 | 0.00419 |
| 6 | 0.00847 | 0.00799 | 0.00672 | 0.00678 |
| 7 | 0.01281 | 0.01283 | 0.01086 | 0.01089 |
| 8 | 0.01991 | 0.02103 | 0.01795 | 0.01776 |
| 9 | 0.03360 | 0.03481 | 0.03256 | 0.03367 |
| 10 | 0.07830 | 0.07790 | 0.09032 | 0.09097 |

Fold 2023

| Group | Baseline: predicted | observed | Deep model: predicted | observed |
|---|---|---|---|---|
| 1 | 0.00061 | 0.00068 | 0.00047 | 0.00047 |
| 2 | 0.00165 | 0.00189 | 0.00120 | 0.00147 |
| 3 | 0.00309 | 0.00324 | 0.00221 | 0.00246 |
| 4 | 0.00506 | 0.00558 | 0.00367 | 0.00407 |
| 5 | 0.00775 | 0.00824 | 0.00577 | 0.00595 |
| 6 | 0.01140 | 0.01161 | 0.00881 | 0.00920 |
| 7 | 0.01666 | 0.01701 | 0.01339 | 0.01404 |
| 8 | 0.02495 | 0.02566 | 0.02085 | 0.02253 |
| 9 | 0.04052 | 0.04116 | 0.03543 | 0.03857 |
| 10 | 0.08489 | 0.07960 | 0.08888 | 0.09918 |
