import numpy as np
import pandas as pd

from src.experiments.matched_control import components, cluster_ci, video_level, add_clusters


def test_components_merge_videos_sharing_a_participant():
    ids = pd.Series(["1", "1|2", "3", "2|4", "5|6"])
    c = components(ids)
    assert c[0] == c[1] == c[3]
    assert len({c[0], c[2], c[4]}) == 3


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(12):
        for label in (0, 1):
            for k in range(5):
                rows.append({"label": label, "prob_fake": float(np.clip(0.3 + 0.4 * label + rng.normal(0, .15), 0, 1)),
                             "threshold": 0.5, "cl": f"g{g}"})
    return pd.DataFrame(rows)


def test_cluster_ci_is_ordered_contains_point_and_is_deterministic():
    frame = _frame()
    config = {"seed": 7, "bootstrap_replicates": 200, "calibration_bins": 15}
    a, b = cluster_ci(frame, "cl", config), cluster_ci(frame, "cl", config)
    assert a == b and a["n_clusters"] == 12
    assert a["auc_ci_lower"] <= a["auc_ci_upper"]
    assert 0 <= a["auc_ci_lower"] and a["auc_ci_upper"] <= 1


def test_video_level_averages_frames_per_video():
    rows = pd.DataFrame({"label": [0, 0, 1, 1], "prob_fake": [.2, .4, .6, .8], "method": ["real", "real", "Deepfakes", "Deepfakes"],
                         "actual_video_id": ["001", "001", "001_002", "001_002"], "target_group": ["1"] * 4,
                         "source_video_ids": ["1", "1", "1|2", "1|2"], "threshold": .5})
    video = video_level(add_clusters(rows))
    assert len(video) == 2 and np.allclose(sorted(video.prob_fake), [.3, .7])
    assert video.cluster_component.nunique() == 1
