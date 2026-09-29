"""Project penultimate-layer detector features into 2D with t-SNE.

The same coordinates can be coloured by class, manipulation method, or source
dataset.  Sampling and dimensionality reduction are deterministic so figures
can be reproduced for the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FaceDataset
from src.data.transforms import build_transform, transform_options
from src.utils import load_checkpoint, resolve_device


def find_classifier(model: nn.Module) -> nn.Linear:
    """Return the final linear classifier whose input is the feature vector."""
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("The model has no linear classifier to hook")
    return linear_layers[-1]


def stratified_sample_indices(
    frame: pd.DataFrame, max_samples: int, seed: int = 42
) -> list[int]:
    """Select a deterministic, approximately balanced sample across methods."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if len(frame) <= max_samples:
        return list(range(len(frame)))
    group_columns = [column for column in ("label", "method") if column in frame]
    if not group_columns:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(len(frame), size=max_samples, replace=False).tolist())

    rng = np.random.default_rng(seed)
    groups: list[list[int]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for _, rows in frame.groupby(group_key, sort=True, dropna=False):
        indices = rows.index.to_numpy(copy=True)
        rng.shuffle(indices)
        groups.append(indices.tolist())

    selected: list[int] = []
    while len(selected) < max_samples and any(groups):
        for indices in groups:
            if indices and len(selected) < max_samples:
                selected.append(indices.pop())
    return sorted(selected)


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module,
    dataset: FaceDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Capture inputs to the final classifier and return features and scores."""
    captured: list[torch.Tensor] = []

    def capture(_: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = find_classifier(model).register_forward_pre_hook(capture)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    row_indices: list[int] = []
    try:
        for images, batch_labels, metadata in loader:
            captured.clear()
            logits = model(images.to(device, non_blocking=True)).flatten()
            if not captured:
                raise RuntimeError("Classifier hook did not capture a feature tensor")
            batch_features = captured[-1]
            if batch_features.ndim > 2:
                batch_features = torch.flatten(batch_features, start_dim=1)
            features.append(batch_features.float().cpu().numpy())
            labels.append(batch_labels.int().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            metadata_indices = metadata["index"]
            row_indices.extend(
                metadata_indices.tolist()
                if isinstance(metadata_indices, torch.Tensor)
                else [int(value) for value in metadata_indices]
            )
    finally:
        handle.remove()
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(probabilities),
        row_indices,
    )


def project_embeddings(
    embeddings: np.ndarray, perplexity: float = 30.0, seed: int = 42
) -> np.ndarray:
    """Standardise, reduce to at most 50 PCA dimensions, then run t-SNE."""
    if embeddings.ndim != 2 or len(embeddings) < 4:
        raise ValueError("t-SNE requires a 2D embedding array with at least four rows")
    scaled = StandardScaler().fit_transform(embeddings)
    components = min(50, scaled.shape[1], scaled.shape[0] - 1)
    reduced = PCA(n_components=components, random_state=seed).fit_transform(scaled)
    effective_perplexity = min(float(perplexity), max(2.0, (len(reduced) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(reduced)


def _scatter_categories(axis: Any, rows: pd.DataFrame, column: str, title: str) -> None:
    categories = sorted(rows[column].fillna("unknown").astype(str).unique())
    colour_map = plt.get_cmap("tab10")
    for index, category in enumerate(categories):
        mask = rows[column].fillna("unknown").astype(str) == category
        axis.scatter(
            rows.loc[mask, "tsne_x"],
            rows.loc[mask, "tsne_y"],
            s=18,
            alpha=0.68,
            color=colour_map(index % 10),
            label=category,
            linewidths=0,
        )
    axis.set_title(title)
    axis.set(xticks=[], yticks=[])
    axis.legend(fontsize=8, frameon=True, markerscale=1.5)


def plot_tsne(rows: pd.DataFrame, output: str | Path, model_name: str) -> None:
    """Create report-ready class, manipulation, and source-domain panels."""
    panels = [("class_name", "Real vs fake"), ("method", "Manipulation method")]
    if rows["source_dataset"].nunique() > 1:
        panels.append(("source_dataset", "Source dataset/domain"))
    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.2), squeeze=False)
    for axis, (column, title) in zip(axes.flat, panels):
        _scatter_categories(axis, rows, column, title)
    figure.suptitle(f"Penultimate-layer t-SNE: {model_name}", fontsize=14)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def generate_tsne(
    checkpoint_path: str | Path,
    csv_paths: list[str | Path],
    data_root: str | Path,
    output: str | Path,
    source_names: list[str] | None = None,
    device_name: str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: int = 1500,
    perplexity: float = 30.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extract, project, save, and describe features from one or more manifests."""
    paths = [Path(path) for path in csv_paths]
    if source_names is not None and len(source_names) != len(paths):
        raise ValueError("source_names must match the number of CSV files")
    names = source_names or [path.stem for path in paths]
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    transform = build_transform(**transform_options(checkpoint.get("config", {})))

    feature_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    quota = max(1, max_samples // len(paths))
    for path, source_name in zip(paths, names):
        dataset = FaceDataset(path, data_root, transform=transform, return_metadata=True)
        indices = stratified_sample_indices(dataset.frame, quota, seed)
        features, labels, probabilities, row_indices = extract_embeddings(
            model, dataset, indices, device, batch_size, num_workers
        )
        metadata = dataset.frame.iloc[row_indices].copy().reset_index(drop=True)
        metadata["label"] = labels
        metadata["prob_fake"] = probabilities
        metadata["source_dataset"] = source_name
        feature_parts.append(features)
        metadata_parts.append(metadata)

    embeddings = np.concatenate(feature_parts)
    rows = pd.concat(metadata_parts, ignore_index=True)
    coordinates = project_embeddings(embeddings, perplexity, seed)
    rows["tsne_x"] = coordinates[:, 0]
    rows["tsne_y"] = coordinates[:, 1]
    rows["class_name"] = rows["label"].map({0: "real", 1: "fake"})

    output = Path(output)
    plot_tsne(rows, output, str(checkpoint["model_name"]))
    csv_output = output.with_suffix(".csv")
    rows.to_csv(csv_output, index=False)
    summary: dict[str, object] = {
        "model": checkpoint["model_name"],
        "checkpoint": str(checkpoint_path),
        "samples": int(len(rows)),
        "feature_dimensions": int(embeddings.shape[1]),
        "perplexity": float(perplexity),
        "seed": int(seed),
        "sources": rows["source_dataset"].value_counts().to_dict(),
        "figure": str(output),
        "coordinates": str(csv_output),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path, nargs="+")
    parser.add_argument("--source-names", nargs="+")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = generate_tsne(
        args.checkpoint,
        args.csv,
        args.data_root,
        args.output,
        args.source_names,
        args.device,
        args.batch_size,
        args.num_workers,
        args.max_samples,
        args.perplexity,
        args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
