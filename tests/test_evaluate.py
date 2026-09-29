import cv2
import numpy as np
import pandas as pd
import torch

from src.evaluate import evaluate_checkpoint
from src.models import build_model


def test_checkpoint_evaluation_smoke(tmp_path):
    rows = []
    for index, label in enumerate([0, 1, 0, 1]):
        image_path = tmp_path / f"image_{index}.jpg"
        cv2.imwrite(str(image_path), np.full((40, 40, 3), 50 + index * 30, dtype=np.uint8))
        rows.append(
            {
                "image_path": image_path.name,
                "label": label,
                "video_id": f"video_{index}",
                "method": "real" if label == 0 else "fake_method",
            }
        )
    csv_path = tmp_path / "test.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    model = build_model("meso", pretrained=False)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_name": "meso",
            "model_options": {"num_classes": 1, "pretrained": False},
            "model_state": model.state_dict(),
            "threshold": 0.5,
            "config": {"training": {"image_size": 64}},
        },
        checkpoint_path,
    )
    result, predictions = evaluate_checkpoint(
        checkpoint_path, csv_path, tmp_path, device_name="cpu", batch_size=2
    )
    assert result["metrics"]["n"] == 4
    assert set(result["per_method"]) == {"fake_method", "real"}
    assert len(predictions) == 4
    assert predictions["prob_fake"].between(0, 1).all()

