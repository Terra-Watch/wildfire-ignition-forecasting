# Tuning level and logistic settings
Decision: boosting and the network are tuned once, at the enriched level on the widest arm's inputs, and the chosen settings are reused at the tabular level. Logistic regularisation stays at the value chosen earlier (C = 0.01).
Why: equal budget for the two models that are tuned, one search per model, and the logistic arms are a fixed reference.
