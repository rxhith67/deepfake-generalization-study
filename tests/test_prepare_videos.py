from pathlib import Path

from src.data.prepare_videos import existing_rows, video_group_id


def test_video_group_id_keeps_manipulated_target_together():
    assert video_group_id("033_097") == "033"
    assert video_group_id("033") == "033"
    assert video_group_id("033_097", "stem") == "033_097"


def test_existing_rows_are_sorted_and_labelled(tmp_path: Path):
    crop_dir = tmp_path / "033_097"
    crop_dir.mkdir()
    (crop_dir / "000001.jpg").touch()
    (crop_dir / "000000.jpg").touch()

    rows = existing_rows(tmp_path, "033_097", 1, "Deepfakes", "ff", "first")

    assert [Path(str(row["image_path"])).name for row in rows] == ["000000.jpg", "000001.jpg"]
    assert {row["video_id"] for row in rows} == {"033"}
    assert {row["label"] for row in rows} == {1}

