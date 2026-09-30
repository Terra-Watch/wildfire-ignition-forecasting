# Burn-free variants for the sealed season
Decision: add a burn-free variant of every arm that uses burn inputs, so the sealed season can be scored without them if the 2026 sources pass the coverage checks.
- Arms without burn inputs already exist: climatology_cell, landcover_season, fwi_season, logistic_weather_static (it is the burn-free version of logistic_with_burn).
- New arms: boosting_no_burn and tabular_network_no_burn, at both information levels. They drop the two burn inputs and, at the enriched level, the four neighbourhood columns built from burn history.
- They are fitted and validated on the same rolling-origin folds as every other arm, so the sealed-season score can be compared with validation scores.
- The specification files are not changed (scoring has begun, and they must not change). The variants are recorded here and in the addendum; their settings live in configs/sealed.yaml.
- The sealed season is declared available for burn-free arms only if FIRMS, ERA5-Land and NDVI for 2026 January-July pass the same coverage checks as the development years. Otherwise it stays unavailable and 2025 is not used in its place.
- Decided before any score of the baseline was written.
Why: the burned-area product ends in 2025-07, so arms with burn inputs cannot be scored on 2026 rows. Burn-free arms can, which keeps the sealed test usable.
