from pathlib import Path

import pandas as pd
import pytest

from src.data.pair_manifests import pair_manifests, path_key


def test_pair_manifests_keeps_only_shared_paths(tmp_path: Path):
    first = tmp_path / "real.csv"
    second = tmp_path / "fake.csv"
    pd.DataFrame(
        {
            "image_path": ["real/10/a.jpg", "real/20/b.jpg"],
            "label": [0, 0],
            "video_id": ["old", "old"],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "image_path": ["fake/10/a.jpg", "fake/30/c.jpg"],
            "label": [1, 1],
            "video_id": ["old", "old"],
        }
    ).to_csv(second, index=False)
    paired = pair_manifests(first, second, tmp_path / "paired.csv")
    assert len(paired) == 2
    assert paired["video_id"].tolist() == ["10/a.jpg", "10/a.jpg"]
    assert path_key("root/class/10/a.jpg") == "10/a.jpg"


def test_pair_manifests_rejects_ambiguous_keys(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"image_path": ["a/x.jpg", "b/x.jpg"], "label": [0, 0]}).to_csv(
        first, index=False
    )
    pd.DataFrame({"image_path": ["c/x.jpg"], "label": [1]}).to_csv(second, index=False)
    with pytest.raises(ValueError, match="not unique"):
        pair_manifests(first, second, tmp_path / "paired.csv", key_depth=1)
