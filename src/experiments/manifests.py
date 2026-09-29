"""Derived source-video-disjoint protocol; never rewrite historical manifests."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path, PurePosixPath

import pandas as pd

from src.experiments.common import configuration, record_run, save_csv, save_json

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
PROTOCOL = "source_video_disjoint_v1"


def canonical_id(path):
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    for root in ("faceforensics", "deepfakeface_sd15", "normalized_external"):
        if root in parts:
            return "/".join(parts[parts.index(root):])
    raise ValueError(f"Unrecognized dataset path: {path}")


def participants(path):
    stem = PurePosixPath(str(path).replace("\\", "/")).parent.name
    tokens = stem.split("_")
    if not 1 <= len(tokens) <= 2 or not all(t.isdigit() for t in tokens):
        raise ValueError(f"Invalid FF++ video stem: {stem}")
    return tuple(str(int(t)) for t in tokens)


def annotate(frame, split):
    frame = frame.copy()
    frame["sample_id"] = frame.image_path.map(canonical_id)
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    frame["target_group"] = frame.image_path.map(lambda p: participants(p)[0])
    if "video_id" in frame and not (frame.video_id.astype(int).astype(str) == frame.target_group).all():
        raise ValueError("Existing target group disagrees with video filename")
    frame["source_video_ids"] = frame.image_path.map(lambda p: "|".join(participants(p)))
    frame["actual_video_id"] = frame.image_path.map(lambda p: PurePosixPath(str(p).replace("\\", "/")).parent.name)
    frame["split"] = split
    frame["protocol_id"] = PROTOCOL
    return frame


def assert_disjoint(parts):
    sample_sets = {k: set(f.sample_id) for k, f in parts.items()}
    source_sets = {k: {s for value in f.source_video_ids for s in value.split("|")} for k, f in parts.items()}
    for a, b in combinations(parts, 2):
        if sample_sets[a] & sample_sets[b] or source_sets[a] & source_sets[b]:
            raise ValueError(f"Sample/source-video leakage: {a} / {b}")
    return {k: len(v) for k, v in source_sets.items()}


def derive(parts):
    annotated = {k: annotate(f, k) for k, f in parts.items()}
    owners = {}
    for split, frame in annotated.items():
        for group in frame.target_group.unique():
            if group in owners and owners[group] != split:
                raise ValueError("Historical target groups overlap")
            owners[group] = split
    retained, excluded = {}, []
    for split, frame in annotated.items():
        keep = frame.source_video_ids.map(lambda value: all(owners.get(x) == split for x in value.split("|")))
        retained[split] = frame.loc[keep].reset_index(drop=True)
        excluded.append(frame.loc[~keep].assign(exclusion_reason="participant_outside_target_partition"))
    assert_disjoint(retained)
    return retained, pd.concat(excluded, ignore_index=True)


def lomo(parts, held_out):
    if held_out not in METHODS:
        raise ValueError("Unknown held-out manipulation")
    result = {s: f.loc[(f.method != held_out) if s != "test" else ((f.label == 0) | (f.method == held_out))].copy()
              for s, f in parts.items()}
    assert_disjoint(result)
    for split, frame in result.items():
        if set(frame.label) != {0, 1}:
            raise ValueError(f"Both classes required in {split}")
        if split != "test" and held_out in set(frame.method):
            raise ValueError("Held-out manipulation leakage")
    return result


def prepare(config):
    root = Path(config["output_dir"]) / "protocol"
    paths = {s: Path(config["data"]["historical_splits"]) / f"{s}.csv" for s in ("train", "val", "test")}
    run = record_run(root, config, paths.values())
    parts, excluded = derive({s: pd.read_csv(p) for s, p in paths.items()})
    for split, frame in parts.items():
        # Portable paths work against local data/processed or the existing Modal archive.
        frame["image_path"] = frame.sample_id
        save_csv(root / "control" / f"{split}.csv", frame)
    for method in METHODS:
        for split, frame in lomo(parts, method).items():
            save_csv(root / f"lomo_{method}" / f"{split}.csv", frame)
    save_csv(root / "excluded.csv", excluded)
    summary = {s: {"images": len(f), "videos": f.groupby(["method", "actual_video_id"]).ngroups,
                   "methods": f.method.value_counts().to_dict()} for s, f in parts.items()}
    save_json(root / "summary.json", {"protocol_id": PROTOCOL, "run_id": run["run_id"],
              "person_identity_disjointness": "not established", "counts": summary,
              "source_groups": assert_disjoint(parts), "excluded_images": len(excluded)})
    return summary


if __name__ == "__main__":
    print(prepare(configuration()))
