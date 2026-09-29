"""Atomic experiment artifacts, hashing, configuration and run records."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config


def configuration(path="config/generalization_upgrade.yaml"):
    config = load_config(path)
    if config["approved_tiers"] != [1, 2]:
        raise ValueError("Only Tier 1 and Tier 2 are approved")
    return config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def save_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def save_csv(path, rows):
    atomic_text(path, rows.to_csv(index=False))


def environment():
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "diffusers", "lpips"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"python": platform.python_version(), "packages": packages,
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short")}


def record_run(directory, config, inputs=()):
    directory = Path(directory)
    record = {"date_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "inputs": {str(p): sha256(p) for p in inputs}, "environment": environment()}
    record["run_id"] = fingerprint({"config": config, "inputs": record["inputs"]})[:16]
    save_json(directory / "run.json", record)
    return record


def update_master(root, rows):
    """Single-writer registry; callers serialize merges after parallel jobs finish."""
    path = Path(root) / "master_results.csv"
    new = pd.DataFrame(rows)
    required = {"experiment", "model", "test_dataset", "condition", "protocol_id", "evaluation_unit"}
    if not required <= set(new):
        raise ValueError(f"Missing result keys: {required - set(new)}")
    if "auc" not in new or new["auc"].isna().any():
        raise ValueError("Only completed two-class evaluations belong in the master table")
    combined = pd.concat([pd.read_csv(path), new], ignore_index=True) if path.exists() else new
    keys = sorted(required | ({"test_manipulation"} if "test_manipulation" in combined else set()))
    save_csv(path, combined.drop_duplicates(keys, keep="last"))
