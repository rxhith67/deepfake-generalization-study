from pathlib import Path

import pandas as pd

from src.data.combine_manifests import combine_manifests


def test_combine_manifests_removes_exact_path_duplicates(tmp_path: Path):
    columns = ["image_path", "label", "video_id", "method", "dataset"]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame([["a.jpg", 0, "a", "real", "ff"]], columns=columns).to_csv(
        first, index=False
    )
    pd.DataFrame(
        [["a.jpg", 0, "a", "real", "ff"], ["b.jpg", 1, "b", "fake", "ff"]],
        columns=columns,
    ).to_csv(second, index=False)

    combined = combine_manifests([first, second])

    assert combined["image_path"].tolist() == ["a.jpg", "b.jpg"]

