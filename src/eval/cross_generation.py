"""Compare in-domain and cross-generation performance for one checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_checkpoint
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--in-domain-csv", required=True, type=Path)
    parser.add_argument("--cross-csv", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/cross_generation.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = (args.checkpoint, args.data_root, args.device, args.batch_size, args.num_workers)
    in_domain, _ = evaluate_checkpoint(common[0], args.in_domain_csv, *common[1:])
    cross, _ = evaluate_checkpoint(common[0], args.cross_csv, *common[1:])
    in_auc = float(in_domain["metrics"]["auc"])
    cross_auc = float(cross["metrics"]["auc"])
    result = {
        "model": in_domain["model"],
        "in_domain": in_domain,
        "cross_generation": cross,
        "auc_drop": in_auc - cross_auc,
    }
    write_json(result, args.output)
    summary = pd.DataFrame(
        [
            {"model": result["model"], "domain": "in_domain", **in_domain["metrics"]},
            {"model": result["model"], "domain": "cross_generation", **cross["metrics"]},
        ]
    ).drop(columns=["confusion"])
    summary["auc_drop"] = [result["auc_drop"], result["auc_drop"]]
    summary.to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"In-domain AUC={in_auc:.4f}; cross-generation AUC={cross_auc:.4f}; drop={result['auc_drop']:.4f}")


if __name__ == "__main__":
    main()

