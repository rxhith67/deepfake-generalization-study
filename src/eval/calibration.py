"""Binary probability calibration and clustered AUC uncertainty."""
import numpy as np
from sklearn.metrics import roc_auc_score


def validate_probabilities(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Nonempty matching one-dimensional vectors required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Binary labels and finite probabilities in [0,1] required")
    return y.astype(int), p


def reliability(labels, probabilities, bins=15):
    y, p = validate_probabilities(labels, probabilities)
    if bins < 1:
        raise ValueError("bins must be positive")
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        mask = assignments == i
        rows.append({"bin": i, "lower": i / bins, "upper": (i + 1) / bins,
                     "count": int(mask.sum()), "mean_probability": float(p[mask].mean()) if mask.any() else None,
                     "fraction_fake": float(y[mask].mean()) if mask.any() else None})
    return rows


def calibration_metrics(labels, probabilities, bins=15):
    y, p = validate_probabilities(labels, probabilities)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    rows = reliability(y, p, bins)
    ece = sum(r["count"] * abs(r["mean_probability"] - r["fraction_fake"]) for r in rows if r["count"]) / len(y)
    return {"ece": float(ece), "nll": float(-(y * np.log(clipped) + (1-y) * np.log1p(-clipped)).mean()),
            "brier": float(np.mean((y-p)**2))}


def bootstrap_auc(labels, probabilities, groups=None, replicates=2000, seed=42, probability_score=True):
    if probability_score:
        y, p = validate_probabilities(labels, probabilities)
    else:
        y, _ = validate_probabilities(labels, np.zeros(len(labels)))
        p = np.asarray(probabilities,dtype=float)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError('Finite matching score vector required')
    if len(np.unique(y)) != 2 or replicates < 1:
        raise ValueError("Two classes and positive replicate count required")
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("Group count mismatch")
    unique, inverse = np.unique(groups, return_inverse=True)
    indices = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(replicates):
        sample = np.concatenate([indices[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[sample])) == 2:
            scores.append(roc_auc_score(y[sample], p[sample]))
    if len(scores) < max(1, replicates // 2):
        raise ValueError("Too few valid two-class bootstrap replicates")
    low, high = np.quantile(scores, [0.025, 0.975])
    return {"auc_ci_lower": float(low), "auc_ci_upper": float(high), "bootstrap_valid": len(scores)}
