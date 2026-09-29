import numpy as np
import pandas as pd
import torch

from src.interpret.tsne import (
    find_classifier,
    project_embeddings,
    stratified_sample_indices,
)
from src.models import build_model


def test_stratified_sample_is_deterministic_and_balanced():
    frame = pd.DataFrame(
        {
            "label": [0] * 20 + [1] * 20 + [1] * 20,
            "method": ["real"] * 20 + ["A"] * 20 + ["B"] * 20,
        }
    )
    first = stratified_sample_indices(frame, 15, seed=7)
    second = stratified_sample_indices(frame, 15, seed=7)
    sampled = frame.iloc[first]
    assert first == second
    assert len(first) == 15
    assert sampled["method"].value_counts().to_dict() == {"real": 5, "A": 5, "B": 5}


def test_projection_returns_finite_two_dimensional_coordinates():
    rng = np.random.default_rng(4)
    coordinates = project_embeddings(rng.normal(size=(30, 12)), perplexity=5, seed=3)
    assert coordinates.shape == (30, 2)
    assert np.isfinite(coordinates).all()


def test_classifier_hook_target_is_last_linear_layer():
    model = build_model("meso", pretrained=False)
    classifier = find_classifier(model)
    assert isinstance(classifier, torch.nn.Linear)
