import pandas as pd

from src.data.subsample_manifest import subsample_groups


def test_subsample_groups_uses_uniform_endpoints():
    frame = pd.DataFrame(
        {
            "image_path": [f"v/{index:02d}.jpg" for index in range(10)],
            "method": ["real"] * 10,
            "video_id": ["v"] * 10,
        }
    )
    selected = subsample_groups(frame, 3, ["method", "video_id"])
    assert selected["image_path"].tolist() == ["v/00.jpg", "v/04.jpg", "v/09.jpg"]

