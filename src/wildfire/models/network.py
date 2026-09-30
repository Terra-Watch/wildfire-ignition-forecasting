"""Tabular network: a small multilayer perceptron trained with design weights and early stopping."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from wildfire import config


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_net(n_inputs: int, width: int, depth: int, dropout: float) -> nn.Sequential:
    """Fully connected layers with ReLU and dropout, one output logit."""
    layers, size = [], n_inputs
    for _ in range(depth):
        layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
        size = width
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


def weighted_log_loss(logits, y, w) -> torch.Tensor:
    """Weighted binary cross-entropy (mean over rows)."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * w).sum() / w.sum()


class Standardiser:
    """Column mean and standard deviation of the fit rows; applies them in place (memory)."""

    def __init__(self, x: np.ndarray):
        self.mu = x.mean(0, dtype=np.float64).astype(np.float32)
        self.sd = (x.std(0, dtype=np.float64) + config.load("models")["logistic"]["scale_epsilon"]).astype(np.float32)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x -= self.mu
        x /= self.sd
        return x


class Network:
    """Fit on standardised inputs with design weights; early stopping on a validation set from the training years."""

    def __init__(self, params: dict, seed: int):
        self.params, self.seed = params, seed

    def _epoch(self, net, opt, data, rng):
        x, y, w = data
        net.train()
        order = rng.permutation(len(y))
        bs = config.load("models")["network"]["batch_size"]
        for i in range(0, len(order), bs):
            idx = torch.as_tensor(order[i:i + bs])
            xb, yb, wb = (t[idx].to(self.device) for t in (x, y, w))
            opt.zero_grad()
            weighted_log_loss(net(xb).squeeze(1), yb, wb).backward()
            opt.step()

    def _logits(self, net, x) -> torch.Tensor:
        net.eval()
        with torch.no_grad():
            return torch.cat([net(x[i:i + 65536].to(self.device)).squeeze(1).cpu() for i in range(0, len(x), 65536)])

    def fit(self, x, y, w, x_val, y_val, w_val):
        """x, x_val: standardised float32 arrays (not copied). Keeps the epoch with the lowest validation loss."""
        cfg = config.load("models")["network"]
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device = pick_device()
        train = (torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), torch.tensor(w / w.mean(), dtype=torch.float32))
        val = (torch.from_numpy(x_val), torch.tensor(y_val, dtype=torch.float32), torch.tensor(w_val / w_val.mean(), dtype=torch.float32))
        p = self.params
        self.net = build_net(x.shape[1], p["width"], p["depth"], p["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        best, best_state, since = np.inf, None, 0
        for _ in range(cfg["max_epochs"]):
            self._epoch(self.net, opt, train, rng)
            loss = float(weighted_log_loss(self._logits(self.net, val[0]), val[1], val[2]))
            if loss < best - 1e-6:
                best, since, best_state = loss, 0, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                since += 1
                if since >= cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.fit_prevalence = float(np.average(y, weights=w))
        self.best_validation_loss = best
        return self

    def predict(self, x):
        """Probabilities for standardised float32 inputs."""
        return torch.sigmoid(self._logits(self.net, torch.from_numpy(x))).numpy().astype(np.float64)
