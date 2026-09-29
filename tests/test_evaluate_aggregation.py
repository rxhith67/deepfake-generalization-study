import pandas as pd

from src.evaluate import aggregate_predictions, binary_method_reports


def test_video_aggregation_and_binary_method_reports():
    rows = pd.DataFrame(
        {
            "dataset": ["ff", "ff", "ff", "ff", "ff", "ff"],
            "method": ["real", "real", "Deepfakes", "Deepfakes", "FaceSwap", "FaceSwap"],
            "video_id": ["a", "a", "b", "b", "c", "c"],
            "label": [0, 0, 1, 1, 1, 1],
            "prob_fake": [0.1, 0.3, 0.7, 0.9, 0.4, 0.6],
        }
    )
    videos = aggregate_predictions(rows, ["dataset", "method", "video_id"])
    assert videos["frames"].tolist() == [2, 2, 2]
    assert videos["prob_fake"].round(2).tolist() == [0.2, 0.8, 0.5]
    reports = binary_method_reports(videos, 0.5)
    assert reports["Deepfakes"]["accuracy"] == 1.0
    assert reports["FaceSwap"]["accuracy"] == 1.0
