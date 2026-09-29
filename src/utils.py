"""Shared reproducibility, device, and checkpoint helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Seeded cuDNN settings provide repeatable runs without forcing unsupported
    # deterministic implementations (which can flood logs or slow training).
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Recreate a registered model and load a project checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_name" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must contain model_name and model_state")
    model_options = dict(checkpoint.get("model_options", {}))
    # Never access the network while loading a trained checkpoint.
    model_options["pretrained"] = False
    model = build_model(checkpoint["model_name"], **model_options)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=True)
