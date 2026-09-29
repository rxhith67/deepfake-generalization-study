import math

from src.eval.metrics import find_optimal_threshold, per_group_reports, report_all


def test_perfect_metrics():
    result = report_all([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["specificity"] == 1.0
    assert result["auc"] == 1.0
    assert result["confusion"] == [[2, 0], [0, 2]]


def test_single_class_auc_is_nan_and_groups_work():
    grouped = per_group_reports([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["real", "real", "fake", "fake"])
    assert math.isnan(grouped["real"]["auc"])
    assert grouped["fake"]["recall"] == 1.0


def test_optimal_threshold_is_derived_from_validation_scores():
    threshold = find_optimal_threshold([0, 0, 1, 1], [0.2, 0.4, 0.45, 0.9])
    result = report_all([0, 0, 1, 1], [0.2, 0.4, 0.45, 0.9], threshold)
    assert threshold == 0.45
    assert result["balanced_accuracy"] == 1.0
