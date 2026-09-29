"""Strict eligibility checks and matched control/LOMO training jobs."""
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.experiments.common import record_run, save_json, sha256
from src.experiments.manifests import METHODS, PROTOCOL, assert_disjoint


def training_config(model, condition, protocol_root, data_root, output_root, device="cuda"):
    if model not in {"hybrid","xception","freq_cnn"}:
        raise ValueError("Only approved architectures")
    if condition != "control" and condition not in {f"lomo_{m}" for m in METHODS}:
        raise ValueError("Only controls and approved LOMO conditions")
    if model == "freq_cnn" and condition != "control":
        raise ValueError("Frequency CNN LOMO is outside the initial bounded run list")
    config = load_config(Path("config") / f"{model}.yaml")
    split_root = Path(protocol_root) / condition
    parts = {s:pd.read_csv(split_root / f"{s}.csv", dtype={"source_video_ids":str}) for s in ("train","val","test")}
    assert_disjoint(parts)
    for split, frame in parts.items():
        if set(frame.protocol_id) != {PROTOCOL} or set(frame.split) != {split}:
            raise ValueError("Protocol/split provenance mismatch")
        if set(frame.label) != {0,1}:
            raise ValueError("Both classes required")
        if condition != "control" and split != "test" and condition.removeprefix("lomo_") in set(frame.method):
            raise ValueError("Held-out manipulation leaked into fitting/selection")
        if any(not (Path(data_root) / p).is_file() for p in frame.image_path):
            raise FileNotFoundError(f"Missing {split} images")
    config["device"] = device
    config["model"]["pretrained"] = model != "freq_cnn"
    config["data"].update({"root":str(data_root),"num_workers":4,
                           **{f"{s}_csv":str(split_root/f'{s}.csv') for s in parts}})
    config["training"].update({"batch_size":64 if model == 'hybrid' else 32,
                                "pin_memory":True,"resumable":True})
    config["training"].pop("init_checkpoint",None)
    config["logging"]["output_dir"] = str(Path(output_root)/condition/model)
    config["experiment"] = {"protocol_id":PROTOCOL,"condition":condition,
                             "initialization":"imagenet" if model != 'freq_cnn' else 'random',
                             "manifest_hashes":{s:sha256(split_root/f'{s}.csv') for s in parts}}
    return config


def run_training(config):
    from src.train import train
    root = Path(config['logging']['output_dir'])
    complete = root / 'complete.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['config'] != config or sha256(saved['checkpoint']) != saved['checkpoint_sha256']:
            raise ValueError('Completed training run provenance mismatch')
        return saved
    record_run(root,config,[config['data'][f'{s}_csv'] for s in ('train','val','test')])
    checkpoint = train(config)
    saved = {'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'config':config}
    save_json(complete,saved)
    return saved
