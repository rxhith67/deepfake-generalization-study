"""Select correct/error cases and render a report-ready Grad-CAM grid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import FaceDataset
from src.data.transforms import build_transform, denormalize_image, transform_options
from src.evaluate import predict
from src.utils import load_checkpoint, resolve_device


class SignedBinaryTarget:
    """Explain evidence for fake (1) or real (0) from a scalar logit."""

    def __init__(self, label: int) -> None:
        self.sign = 1.0 if label == 1 else -1.0

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.sign * model_output.flatten()[0]


def find_last_conv(model: nn.Module) -> nn.Conv2d:
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise ValueError("No Conv2d target layer found; specify a CNN-based registered model")
    return layers[-1]


def select_cases(labels: list[int], probabilities: list[float], per_category: int, threshold: float) -> list[tuple[int, str]]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    categories = {
        "correct real": lambda y, p: y == 0 and p == 0,
        "correct fake": lambda y, p: y == 1 and p == 1,
        "false positive": lambda y, p: y == 0 and p == 1,
        "false negative": lambda y, p: y == 1 and p == 0,
    }
    selected = []
    for name, predicate in categories.items():
        matches = [i for i, pair in enumerate(zip(labels, predictions)) if predicate(*pair)]
        selected.extend((index, name) for index in matches[:per_category])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples-per-category", type=int, default=2)
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as error:
        raise SystemExit("Install the 'grad-cam' package from requirements.txt") from error
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    transform_kwargs = transform_options(checkpoint.get("config", {}))
    dataset = FaceDataset(
        args.csv, args.data_root, build_transform(**transform_kwargs), return_metadata=True
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    labels, probabilities, _ = predict(model, loader, device)
    cases = select_cases(labels, probabilities, args.samples_per_category, threshold)
    if not cases:
        raise SystemExit("No qualifying samples found")
    target_layer = find_last_conv(model)
    columns = min(4, len(cases))
    rows = math.ceil(len(cases) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for axis, (index, category) in zip(axes.flat, cases):
            tensor, label, metadata = dataset[index]
            batch = tensor.unsqueeze(0).to(device)
            grayscale = cam(input_tensor=batch, targets=[SignedBinaryTarget(int(label.item()))])[0]
            overlay = show_cam_on_image(
                denormalize_image(tensor, transform_kwargs["mean"], transform_kwargs["std"]),
                grayscale,
                use_rgb=True,
            )
            axis.imshow(overlay)
            axis.set_title(f"{category}\ny={int(label.item())}, p(fake)={probabilities[index]:.3f}")
            axis.axis("off")
        for axis in axes.flat[len(cases) :]:
            axis.axis("off")
    figure.suptitle(f"Grad-CAM: {checkpoint['model_name']}")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {len(cases)} Grad-CAM examples to {args.output}")


if __name__ == "__main__":
    main()
