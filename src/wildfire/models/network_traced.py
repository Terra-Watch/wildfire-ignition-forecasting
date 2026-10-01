"""The tabular network with a per-epoch history, for the walk-through notebook only (not used by any official step)."""
from __future__ import annotations

import numpy as np
import torch

from wildfire import config
from wildfire.models.network import Network, build_net, pick_device, weighted_log_loss


class TracedNetwork(Network):
    """Same training loop as `Network.fit` (same seeding, optimiser, loss, early stopping, best-epoch restore), plus the
    design-weighted loss of a probe sample of the training rows and of the early-stopping rows after every epoch.
    Its predictions must equal those of `Network` for the same seed; the notebook checks this against the saved run."""

    def fit(self, x, y, w, x_val, y_val, w_val, x_probe, y_probe, w_probe):
        cfg = config.load("models")["network"]
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device = pick_device()
        make = lambda a, b, c: (torch.from_numpy(a), torch.tensor(b, dtype=torch.float32), torch.tensor(c / c.mean(), dtype=torch.float32))
        train, val, probe = make(x, y, w), make(x_val, y_val, w_val), make(x_probe, y_probe, w_probe)
        p = self.params
        self.net = build_net(x.shape[1], p["width"], p["depth"], p["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        best, best_state, since, self.history = np.inf, None, 0, []
        for epoch in range(cfg["max_epochs"]):
            self._epoch(self.net, opt, train, rng)
            stop_loss = float(weighted_log_loss(self._logits(self.net, val[0]), val[1], val[2]))
            probe_loss = float(weighted_log_loss(self._logits(self.net, probe[0]), probe[1], probe[2]))
            self.history.append(dict(epoch=epoch + 1, train_loss=probe_loss, stop_year_loss=stop_loss))
            if stop_loss < best - cfg["min_improvement"]:
                best, since, best_state = stop_loss, 0, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                since += 1
                if since >= cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.fit_prevalence = float(np.average(y, weights=w))
        self.best_validation_loss = best
        return self
