"""Binary-classification metrics used throughout the experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold(
    y_true: Iterable[int | float], y_prob: Iterable[float]
) -> float:
    """Choose a validation-only threshold that maximises balanced accuracy."""
    truth = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(y_prob), dtype=float)
    if truth.size == 0 or probability.size != truth.size or np.unique(truth).size != 2:
        raise ValueError("Threshold optimisation requires two non-empty classes")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(truth, probability)
    scores = (true_positive_rate + (1.0 - false_positive_rate)) / 2.0
    finite = np.isfinite(thresholds)
    candidates = np.flatnonzero(finite & np.isclose(scores, np.max(scores[finite])))
    # Stable tie-break: prefer the equally good threshold closest to the
    # conventional 0.5 operating point.
    index = candidates[np.argmin(np.abs(thresholds[candidates] - 0.5))]
    return float(np.clip(thresholds[index], 0.0, 1.0))


def report_all(
    y_true: Iterable[int | float], y_prob: Iterable[float], threshold: float = 0.5
) -> dict[str, object]:
    truth = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(y_prob), dtype=float)
    if truth.size == 0 or probability.size != truth.size:
        raise ValueError("y_true and y_prob must be non-empty and have equal length")
    prediction = (probability >= threshold).astype(int)
    auc = float(roc_auc_score(truth, probability)) if np.unique(truth).size == 2 else float("nan")
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "n": int(truth.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(truth, prediction))
            if np.unique(truth).size == 2
            else float("nan")
        ),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(1.0 - specificity),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "auc": auc,
        "confusion": matrix.tolist(),
    }


def per_group_reports(
    y_true: Iterable[int | float],
    y_prob: Iterable[float],
    groups: Iterable[str],
    threshold: float = 0.5,
) -> dict[str, dict[str, object]]:
    truth = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(y_prob), dtype=float)
    group_array = np.asarray(list(groups), dtype=str)
    if not (len(truth) == len(probability) == len(group_array)):
        raise ValueError("truth, probability, and groups must have equal length")
    return {
        group: report_all(truth[group_array == group], probability[group_array == group], threshold)
        for group in sorted(set(group_array))
    }
