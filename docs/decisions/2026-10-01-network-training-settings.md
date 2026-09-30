# Network training settings
Decision: batch size 4096, at most 20 epochs, stop after 2 epochs without an improvement of 1e-6 in the weighted validation log loss, keep the best epoch, AdamW with its default moment settings, inputs standardised with the mean and deviation of the training rows. All values live in configs/models.yaml.
Why: the specification fixes the architecture, optimiser, loss, early-stopping criterion and seeds; these are the remaining settings, chosen once for cost and not tuned.
