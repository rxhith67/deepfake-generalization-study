"""Train a configured binary deepfake detector."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import FaceDataset, make_balanced_sampler
from src.data.transforms import build_transform, transform_options
from src.eval.metrics import find_optimal_threshold, report_all
from src.models import build_model
from src.models.registry import freeze_first_fraction, unfreeze_all
from src.utils import resolve_device, seed_everything


def make_loaders(config: dict[str, Any], device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader]:
    data = config["data"]
    training = config["training"]
    transform_kwargs = transform_options(config)
    train_set = FaceDataset(
        data["train_csv"],
        data["root"],
        build_transform(train=True, augmentation=config.get("augmentation"), **transform_kwargs),
    )
    val_set = FaceDataset(data["val_csv"], data["root"], build_transform(**transform_kwargs))
    test_set = FaceDataset(data["test_csv"], data["root"], build_transform(**transform_kwargs))
    sampler = make_balanced_sampler(train_set) if training.get("balanced_sampling", True) else None
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(data.get("num_workers", 4)),
        "pin_memory": bool(training.get("pin_memory", device.type == "cuda")),
    }
    train_loader = DataLoader(
        train_set, sampler=sampler, shuffle=sampler is None, drop_last=False, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> AdamW:
    training = config["training"]
    model_name = config["model"]["name"]
    base_lr = float(training["lr"])
    weight_decay = float(training["weight_decay"])
    if model_name not in {"xception", "hybrid"}:
        return AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    head_tokens = ("head", "classifier", "fc")
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        (head if any(token in name.lower() for token in head_tokens) else backbone).append(parameter)
    return AdamW(
        [
            {"params": backbone, "lr": float(training.get("backbone_lr", 1e-5))},
            {"params": head, "lr": base_lr},
        ],
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images).flatten()
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach()) * len(labels)
        sample_count += len(labels)
    return loss_sum / max(sample_count, 1)


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimize_threshold: bool = False,
) -> dict[str, object]:
    model.eval()
    labels_all: list[int] = []
    probabilities: list[float] = []
    loss_sum = 0.0
    sample_count = 0
    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(images).flatten()
        loss_sum += float(criterion(logits, labels_device)) * len(labels)
        sample_count += len(labels)
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels_all.extend(labels.int().tolist())
    operating_threshold = (
        find_optimal_threshold(labels_all, probabilities) if optimize_threshold else threshold
    )
    metrics = report_all(labels_all, probabilities, operating_threshold)
    metrics["loss"] = loss_sum / max(sample_count, 1)
    return metrics


def train(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    model_config = config["model"]
    model_name = str(model_config["name"])
    pretrained = bool(model_config.get("pretrained", False))
    model = build_model(model_name, num_classes=1, pretrained=pretrained).to(device)
    initial_checkpoint_path = config["training"].get("init_checkpoint")
    initial_checkpoint: dict[str, Any] | None = None
    initial_epoch = 0
    if initial_checkpoint_path:
        initial_checkpoint = torch.load(
            Path(initial_checkpoint_path), map_location=device, weights_only=False
        )
        if initial_checkpoint.get("model_name") != model_name:
            raise ValueError(
                "Initial checkpoint model does not match configured model: "
                f"{initial_checkpoint.get('model_name')} != {model_name}"
            )
        model.load_state_dict(initial_checkpoint["model_state"])
        initial_epoch = int(initial_checkpoint.get("epoch", 0))
        print(f"Resuming weights from {initial_checkpoint_path} (epoch {initial_epoch})")
    freeze_epochs = int(model_config.get("freeze_epochs", 0))
    if freeze_epochs:
        freeze_first_fraction(model, float(model_config.get("freeze_fraction", 0.60)))

    train_loader, val_loader, test_loader = make_loaders(config, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = make_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except AttributeError:  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    threshold = float(config["training"].get("threshold", 0.5))
    patience = int(config["training"].get("early_stopping_patience", 3))

    output_dir = Path(config["logging"]["output_dir"])
    checkpoint_path = output_dir / "checkpoints" / f"{model_name}_best.pt"
    log_path = output_dir / "logs" / f"{model_name}_epochs.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_auc = -math.inf
    stale_epochs = 0
    start_epoch = 0
    # Opt-in only: historical training behavior and artifacts are unchanged.
    resumable = bool(config["training"].get("resumable", False))
    resume_path = output_dir / "checkpoints" / "training_state_last.pt"
    if initial_checkpoint is not None:
        initial_validation = initial_checkpoint.get("validation", {})
        initial_auc = float(initial_validation.get("auc", -math.inf))
        if math.isfinite(initial_auc):
            best_auc = initial_auc
            retained = {
                **initial_checkpoint,
                "config": config,
                "resumed_from": str(initial_checkpoint_path),
            }
            torch.save(retained, checkpoint_path)
            print(f"Retained initial validation AUC={initial_auc:.4f} as refinement baseline")

    if resumable and resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume["config"] != config:
            raise ValueError("Resume configuration differs from saved run")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch, best_auc, stale_epochs = resume["next_epoch"], resume["best_auc"], resume["stale_epochs"]
        history = resume["history"]
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng"]])
        if start_epoch >= freeze_epochs:
            unfreeze_all(model)
        print(f"Resumed complete training state at epoch {start_epoch}")
    print(f"Training {model_name} on {device}; {len(train_loader.dataset)} train images")
    for epoch in range(start_epoch, epochs):
        if stale_epochs >= patience:
            break
        if freeze_epochs and epoch == freeze_epochs:
            unfreeze_all(model)
            print(f"Epoch {epoch + 1}: unfroze transfer backbone")
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, amp_enabled)
        validation = evaluate_loader(
            model, val_loader, criterion, device, threshold, optimize_threshold=True
        )
        scheduler.step()
        auc = float(validation["auc"])
        score = auc if math.isfinite(auc) else -float(validation["loss"])
        row = {
            "epoch": initial_epoch + epoch + 1,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in validation.items() if key != "confusion"},
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(log_path, index=False)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} val_auc={auc:.4f} val_f1={validation['f1']:.4f}"
        )
        if score > best_auc:
            best_auc = score
            stale_epochs = 0
            torch.save(
                {
                    "format_version": 1,
                    "model_name": model_name,
                    "model_options": {"num_classes": 1, "pretrained": pretrained},
                    "model_state": model.state_dict(),
                    "epoch": initial_epoch + epoch + 1,
                    "validation": validation,
                    "threshold": float(validation["threshold"]),
                    "seed": seed,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if resumable:
            temporary = resume_path.with_suffix(".tmp")
            torch.save({"config": config, "model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "next_epoch": epoch + 1,
                        "best_auc": best_auc, "stale_epochs": stale_epochs, "history": history,
                        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, temporary)
            os.replace(temporary, resume_path)
        if stale_epochs >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_loader(
        model, test_loader, criterion, device, float(checkpoint.get("threshold", threshold))
    )
    checkpoint["test"] = test_metrics
    torch.save(checkpoint, checkpoint_path)
    print(f"Best checkpoint: {checkpoint_path}\nTest metrics: {test_metrics}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", help="Override config device (e.g. cpu, cuda, cuda:0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    train(config)


if __name__ == "__main__":
    main()
