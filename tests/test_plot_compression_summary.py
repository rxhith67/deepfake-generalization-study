from pathlib import Path

import pandas as pd

from src.eval.plot_compression_summary import plot_compression_summary


def test_plot_compression_summary_combines_models(tmp_path: Path):
    inputs = []
    for model in ("a", "b"):
        path = tmp_path / f"{model}.csv"
        pd.DataFrame(
            {
                "model": [model, model],
                "quality": [100, 10],
                "accuracy": [0.8, 0.6],
                "auc": [0.9, 0.7],
            }
        ).to_csv(path, index=False)
        inputs.append(path)
    output = tmp_path / "summary.png"
    table = plot_compression_summary(inputs, output)
    assert len(table) == 4
    assert output.is_file() and output.stat().st_size > 0
