"""Train the 8-seed ensemble on one fold of a model input package and write each seed's validation logits and weights. Validation years 2024 and 2023 only.
Usage: python scripts/train_ensemble.py --package <package folder> --year 2024 --output <folder> [--device auto|cpu|cuda] [--seeds 0 1 2]
A real fold needs a GPU for a practical run time (about 12 minutes per seed on one 24 GB GPU); the synthetic package runs on a CPU in seconds."""
import argparse
from pathlib import Path

import numpy as np

from deep_multimodal.data import load_fold
from deep_multimodal.settings import load_settings
from deep_multimodal.training import TrainingSettings, pick_device, train_ensemble

parser = argparse.ArgumentParser()
parser.add_argument("--package", required=True)
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", default="auto")
parser.add_argument("--seeds", type=int, nargs="*")
parser.add_argument("--config")
parser.add_argument("--synthetic", action="store_true", help="allow a synthetic validation year such as 2022")
args = parser.parse_args()
if args.year not in (2023, 2024) and not args.synthetic:
    raise SystemExit("only validation years 2024 and 2023 are scored")
model_settings, training, _ = load_settings(*( [args.config] if args.config else []))
if args.seeds:
    training = TrainingSettings(**{**training.__dict__, "seeds": tuple(args.seeds)})
device = pick_device(args.device)
fold = load_fold(Path(args.package), args.year, device)
ensemble = train_ensemble(fold, model_settings, training, device, Path(args.output))
np.save(Path(args.output) / "ensemble_validation_probability.npy", ensemble)
print("ensemble probabilities written for", len(ensemble), "validation rows")
