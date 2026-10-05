"""Training: two stages per seed, then an ensemble of seeds.

Stage 1 trains on the fit years before the last fit year and stops when the prior-shift-corrected log loss on the last fit year stops improving (patience 2, at most 20 epochs);
the best epoch count is kept. Stage 2 retrains from the same seed on all fit rows for exactly that many epochs. The ensemble is the mean of the seeds' probabilities. Hard labels only.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from deep_multimodal.data import FoldData, SplitTensors
from deep_multimodal.metrics import stopping_score
from deep_multimodal.model import DeepMultimodalModel, ModelSettings, auxiliary_loss


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 20
    patience: int = 2
    minimum_gain: float = 1e-6
    batch_size: int = 4096
    seeds: tuple = tuple(range(8))


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def predict_logits(model: DeepMultimodalModel, data: SplitTensors, batch_size: int = 8192) -> np.ndarray:
    """Logits of every row of `data`, in the order of the rows."""
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
        for start in range(0, data.size, batch_size):
            index = torch.arange(start, min(start + batch_size, data.size), device=data.device)
            out.append(model(data.batch(index))[0].float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def _train(data: SplitTensors, settings: ModelSettings, training: TrainingSettings, seed: int, device: torch.device, epochs: int, after_epoch=None) -> DeepMultimodalModel:
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = DeepMultimodalModel(settings).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    for epoch in range(epochs):
        model.train()
        order = torch.from_numpy(generator.permutation(data.size)).to(device)
        for start in range(0, data.size, training.batch_size):
            batch = data.batch(order[start:start + training.batch_size])
            optimiser.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logit, prediction = model(batch, training=True)
            loss = functional.binary_cross_entropy_with_logits(logit.float(), batch["label"]) + auxiliary_loss({k: v.float() for k, v in prediction.items()}, batch, settings.auxiliary_weight)
            loss.backward()
            optimiser.step()
        if after_epoch is not None and after_epoch(model, epoch):
            break
    return model


def stopping_arguments(meta: dict, stop: SplitTensors) -> dict:
    """True fire rate of the stopping year's own unthinned rows, and the weight that stands in for its thinned negatives."""
    if "stop_year_pool_rows" not in meta:
        return {}
    pool_negatives = meta["stop_year_pool_rows"] - meta["stop_year_pool_onsets"]
    return dict(true_rate=meta["stop_year_pool_onsets"] / meta["stop_year_pool_rows"], negative_weight=pool_negatives / float((stop.label.cpu().numpy() == 0).sum()))


def train_two_stage(fold: FoldData, settings: ModelSettings, training: TrainingSettings, seed: int, device: torch.device) -> tuple[DeepMultimodalModel, int]:
    """Returns the final model (trained on all fit rows) and the epoch count chosen in stage 1."""
    stop_labels = fold.stop.label.cpu().numpy()
    arguments = stopping_arguments(fold.meta, fold.stop)
    state = dict(best=np.inf, average_precision=-1.0, since=0, epoch=0, weights=None)

    def after_epoch(model, epoch):
        probability = 1 / (1 + np.exp(-predict_logits(model, fold.stop)))
        loss, precision = stopping_score(probability, stop_labels, fold.meta["fit_prevalence"], arguments.get("true_rate"), arguments.get("negative_weight", 1.0))
        if loss < state["best"] - training.minimum_gain or (abs(loss - state["best"]) <= training.minimum_gain and precision > state["average_precision"]):
            state.update(best=loss, average_precision=precision, since=0, epoch=epoch + 1, weights={k: v.detach().clone() for k, v in model.state_dict().items()})
        else:
            state["since"] += 1
        return state["since"] >= training.patience

    _train(fold.train, settings, training, seed, device, training.max_epochs, after_epoch)
    final = _train(fold.fit_all, settings, training, seed, device, state["epoch"])
    return final, state["epoch"]


def train_ensemble(fold: FoldData, settings: ModelSettings, training: TrainingSettings, device: torch.device, output_dir: Path | None = None, log=print) -> np.ndarray:
    """Train every seed, write each seed's logits (and weights, when `output_dir` is given), and return the ensemble probability of the validation rows."""
    probabilities = []
    for seed in training.seeds:
        started = time.time()
        model, epochs = train_two_stage(fold, settings, training, seed, device)
        logits = predict_logits(model, fold.validation)
        probabilities.append(1 / (1 + np.exp(-logits)))
        log(f"seed {seed}: {epochs} epochs, {time.time() - started:.0f} s")
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / f"seed{seed}_validation_logits.npy", logits)
            torch.save(model.state_dict(), output_dir / f"seed{seed}_weights.pt")
            (output_dir / f"seed{seed}.json").write_text(json.dumps(dict(seed=seed, epochs=epochs, parameters=int(sum(p.numel() for p in model.parameters())))))
    return np.mean(probabilities, 0)
