"""Loading trained weights, from a file written by this code (a plain state dictionary) or by the research code (a training checkpoint with the state under the key "model")."""
from __future__ import annotations

from pathlib import Path

import torch

from deep_multimodal.model import DeepMultimodalModel, ModelSettings, convert_legacy_state_dict


def load_weights(path: Path | str, settings: ModelSettings = ModelSettings()) -> DeepMultimodalModel:
    """A model in evaluation mode with the weights of `path`. Weight files are not part of the repository (see docs/MODEL.md)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = DeepMultimodalModel(settings)
    model.load_state_dict(state if "fire_channel_projection.weight" in state else convert_legacy_state_dict(state))
    return model.eval()
