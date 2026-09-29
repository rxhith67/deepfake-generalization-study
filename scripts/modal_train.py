"""Train the strongest detectors on a Modal GPU and persist their outputs."""

from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT = "/root/project"
VOLUME_ROOT = "/mnt/deepfake"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "numpy==1.26.4",
        "torch==2.2.2",
        "torchvision==0.17.2",
        "timm==1.0.7",
        "opencv-python-headless==4.10.0.84",
        "albumentations==1.3.1",
        "scikit-learn==1.5.1",
        "pandas==2.2.2",
        "pyyaml==6.0.2",
        "tqdm==4.66.5",
        "huggingface-hub==0.34.4",
    )
    .add_local_dir(PROJECT_ROOT / "src", remote_path=f"{REMOTE_PROJECT}/src", copy=True)
    .add_local_dir(PROJECT_ROOT / "config", remote_path=f"{REMOTE_PROJECT}/config", copy=True)
)

app = modal.App("deepfake-detection-training", image=image)
volume = modal.Volume.from_name("deepfake-detection")


@app.function(
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def train_detector(model_name: str) -> dict[str, object]:
    import os
    import shutil
    import subprocess
    import sys

    sys.path.insert(0, str(REMOTE_PROJECT))
    os.chdir(REMOTE_PROJECT)

    from src.config import load_config
    from src.evaluate import evaluate_checkpoint
    from src.train import train
    from src.utils import write_json

    if model_name not in {"hybrid", "xception"}:
        raise ValueError("model_name must be hybrid or xception")

    remote_project = Path(REMOTE_PROJECT)
    volume_root = Path(VOLUME_ROOT)
    data_root = Path("/tmp/deepfake_data")
    data_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tar",
            "-xf",
            str(volume_root / "data" / "modal_faceforensics.tar"),
            "-C",
            str(data_root),
        ],
        check=True,
    )
    config = load_config(remote_project / "config" / f"{model_name}_200video.yaml")
    split_root = volume_root / "data" / "splits_200video"
    config["data"].update(
        {
            "root": str(data_root),
            "train_csv": str(split_root / "train.csv"),
            "val_csv": str(split_root / "val.csv"),
            "test_csv": str(split_root / "test.csv"),
            "num_workers": 4,
        }
    )
    config["device"] = "cuda"
    config["training"]["pin_memory"] = True
    config["training"]["batch_size"] = 64 if model_name == "hybrid" else 32
    output_dir = volume_root / "results" / f"{model_name}_200video_full"
    config["logging"]["output_dir"] = str(output_dir)

    checkpoint_path = train(config)
    archived = output_dir / "checkpoints" / f"{model_name}_200video_full_best.pt"
    shutil.copy2(checkpoint_path, archived)
    result, predictions = evaluate_checkpoint(
        archived,
        split_root / "test.csv",
        data_root,
        "cuda",
        config["training"]["batch_size"],
        config["data"]["num_workers"],
    )
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    write_json(result, table_dir / f"{model_name}_test.json")
    predictions.to_csv(table_dir / f"{model_name}_test_predictions.csv", index=False)
    volume.commit()
    return {
        "model": model_name,
        "checkpoint": str(archived.relative_to(volume_root)),
        "metrics": result["metrics"],
        "video_metrics": result.get("video_metrics"),
    }


@app.function(
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def refine_hybrid_detector() -> dict[str, object]:
    """Continue the best hybrid checkpoint at a lower learning rate."""
    import os
    import shutil
    import subprocess
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_PROJECT))
    os.chdir(REMOTE_PROJECT)

    from src.config import load_config
    from src.evaluate import evaluate_checkpoint
    from src.train import train
    from src.utils import write_json

    volume_root = Path(VOLUME_ROOT)
    data_root = Path("/tmp/deepfake_data")
    data_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tar",
            "-xf",
            str(volume_root / "data" / "modal_faceforensics.tar"),
            "-C",
            str(data_root),
        ],
        check=True,
    )
    split_root = volume_root / "data" / "splits_200video"
    source_checkpoint = (
        volume_root
        / "results"
        / "hybrid_200video_full"
        / "checkpoints"
        / "hybrid_200video_full_best.pt"
    )
    output_dir = volume_root / "results" / "hybrid_200video_refined"
    config = load_config(Path(REMOTE_PROJECT) / "config" / "hybrid_200video.yaml")
    config["data"].update(
        {
            "root": str(data_root),
            "train_csv": str(split_root / "train.csv"),
            "val_csv": str(split_root / "val.csv"),
            "test_csv": str(split_root / "test.csv"),
            "num_workers": 4,
        }
    )
    config["device"] = "cuda"
    config["model"]["freeze_epochs"] = 0
    config["training"].update(
        {
            "init_checkpoint": str(source_checkpoint),
            "epochs": 5,
            "early_stopping_patience": 2,
            "batch_size": 64,
            "pin_memory": True,
            "lr": 2.5e-5,
            "backbone_lr": 2.5e-6,
        }
    )
    config["logging"]["output_dir"] = str(output_dir)

    checkpoint_path = train(config)
    archived = output_dir / "checkpoints" / "hybrid_200video_refined_best.pt"
    shutil.copy2(checkpoint_path, archived)
    result, predictions = evaluate_checkpoint(
        archived, split_root / "test.csv", data_root, "cuda", 64, 4
    )
    val_result, val_predictions = evaluate_checkpoint(
        archived, split_root / "val.csv", data_root, "cuda", 64, 4
    )
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    write_json(result, table_dir / "hybrid_test.json")
    predictions.to_csv(table_dir / "hybrid_test_predictions.csv", index=False)
    write_json(val_result, table_dir / "hybrid_val.json")
    val_predictions.to_csv(table_dir / "hybrid_val_predictions.csv", index=False)
    volume.commit()
    return {
        "checkpoint": str(archived.relative_to(volume_root)),
        "epoch": int(torch.load(archived, map_location="cpu", weights_only=False)["epoch"]),
        "metrics": result["metrics"],
        "video_metrics": result.get("video_metrics"),
        "validation_metrics": val_result["metrics"],
    }


@app.function(
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def evaluate_saved_detector(model_name: str, split: str = "val") -> dict[str, object]:
    """Evaluate a completed cloud checkpoint and persist predictions for calibration."""
    import os
    import subprocess
    import sys

    sys.path.insert(0, str(REMOTE_PROJECT))
    os.chdir(REMOTE_PROJECT)

    from src.evaluate import evaluate_checkpoint
    from src.utils import write_json

    if model_name not in {"hybrid", "xception"}:
        raise ValueError("model_name must be hybrid or xception")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")

    volume_root = Path(VOLUME_ROOT)
    data_root = Path("/tmp/deepfake_data")
    data_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tar",
            "-xf",
            str(volume_root / "data" / "modal_faceforensics.tar"),
            "-C",
            str(data_root),
        ],
        check=True,
    )
    output_dir = volume_root / "results" / f"{model_name}_200video_full"
    checkpoint = output_dir / "checkpoints" / f"{model_name}_200video_full_best.pt"
    split_csv = volume_root / "data" / "splits_200video" / f"{split}.csv"
    result, predictions = evaluate_checkpoint(
        checkpoint,
        split_csv,
        data_root,
        "cuda",
        64 if model_name == "hybrid" else 32,
        4,
    )
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = table_dir / f"{model_name}_{split}.json"
    predictions_path = table_dir / f"{model_name}_{split}_predictions.csv"
    write_json(result, metrics_path)
    predictions.to_csv(predictions_path, index=False)
    volume.commit()
    return {
        "model": model_name,
        "split": split,
        "metrics": result["metrics"],
        "video_metrics": result.get("video_metrics"),
    }


@app.function(
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def evaluate_saved_robustness(model_name: str) -> list[dict[str, object]]:
    """Measure JPEG degradation for a completed checkpoint on the full test split."""
    import os
    import subprocess
    import sys

    import pandas as pd

    sys.path.insert(0, str(REMOTE_PROJECT))
    os.chdir(REMOTE_PROJECT)

    from src.evaluate import evaluate_checkpoint
    from src.utils import write_json

    if model_name not in {"hybrid", "xception"}:
        raise ValueError("model_name must be hybrid or xception")

    volume_root = Path(VOLUME_ROOT)
    data_root = Path("/tmp/deepfake_data")
    data_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tar",
            "-xf",
            str(volume_root / "data" / "modal_faceforensics.tar"),
            "-C",
            str(data_root),
        ],
        check=True,
    )
    output_dir = volume_root / "results" / f"{model_name}_200video_full"
    checkpoint = output_dir / "checkpoints" / f"{model_name}_200video_full_best.pt"
    test_csv = volume_root / "data" / "splits_200video" / "test.csv"
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for quality in (100, 90, 60, 40, 20, 10):
        result, _ = evaluate_checkpoint(
            checkpoint,
            test_csv,
            data_root,
            "cuda",
            64 if model_name == "hybrid" else 32,
            4,
            jpeg_quality=quality,
        )
        metrics = result["metrics"]
        rows.append(
            {
                "model": model_name,
                "quality": quality,
                **{key: value for key, value in metrics.items() if key != "confusion"},
            }
        )
        details[str(quality)] = result
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_dir / f"{model_name}_compression.csv", index=False)
    write_json(details, table_dir / f"{model_name}_compression.json")
    volume.commit()
    return rows


@app.local_entrypoint()
def main(
    model: str = "hybrid",
    evaluate_only: bool = False,
    robustness: bool = False,
    refine_hybrid: bool = False,
    split: str = "val",
) -> None:
    if refine_hybrid:
        print(refine_hybrid_detector.remote())
    elif robustness:
        print(evaluate_saved_robustness.remote(model))
    elif evaluate_only:
        print(evaluate_saved_detector.remote(model, split))
    else:
        print(train_detector.remote(model))
