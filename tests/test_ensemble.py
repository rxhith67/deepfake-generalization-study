from pathlib import Path

import pandas as pd

from src.eval.ensemble import combine_predictions, detailed_report


def _write_predictions(path: Path, probabilities: list[float]) -> None:
    pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "label": [0, 0, 1, 1],
            "dataset": ["ff"] * 4,
            "method": ["real", "real", "Deepfakes", "Deepfakes"],
            "video_id": ["real", "real", "fake", "fake"],
            "prob_fake": probabilities,
            "prediction": [int(value >= 0.5) for value in probabilities],
        }
    ).to_csv(path, index=False)


def test_weighted_ensemble_preserves_metadata_and_video_metrics(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_predictions(first, [0.1, 0.2, 0.8, 0.9])
    _write_predictions(second, [0.3, 0.4, 0.6, 0.7])

    rows = combine_predictions([first, second], [0.75, 0.25])

    assert rows["prob_fake"].round(3).tolist() == [0.15, 0.25, 0.75, 0.85]
    assert rows["method"].tolist() == ["real", "real", "Deepfakes", "Deepfakes"]
    report = detailed_report(rows, 0.5)
    assert report["metrics"]["accuracy"] == 1.0
    assert report["video_metrics"]["accuracy"] == 1.0

