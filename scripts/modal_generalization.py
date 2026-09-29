"""Bounded Tier-1 source-video-disjoint training: 3 controls, then 8 LOMO runs.

Run only after local Tier-1 baseline/bias/calibration stages pass:
  python -m modal run --detach scripts/modal_generalization.py
Artifacts remain in /generalization_upgrade on the existing approved volume.
"""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("numpy==1.26.4","torch==2.2.2","torchvision==0.17.2","timm==1.0.7",
                         "opencv-python-headless==4.10.0.84","albumentations==1.3.1","scikit-learn==1.5.1",
                         "pandas==2.2.2","pyyaml==6.0.2","tqdm==4.66.5","huggingface-hub==0.34.4",
                         "matplotlib==3.9.2")
         .add_local_dir(ROOT/"src",remote_path="/root/project/src",copy=True)
         .add_local_dir(ROOT/"config",remote_path="/root/project/config",copy=True)
         .add_local_dir(ROOT/"outputs/generalization_upgrade/protocol",remote_path="/root/protocol",copy=True))
app = modal.App("deepfake-generalization-upgrade",image=image)
volume = modal.Volume.from_name("deepfake-detection")


@app.function(gpu="A10",cpu=8,memory=32768,timeout=1800,max_containers=1,
              volumes={"/mnt/deepfake":volume})
def train_job(model: str,condition: str):
    import os
    import sys
    import tarfile
    sys.path.insert(0,"/root/project")
    os.chdir("/root/project")
    from src.experiments.training import training_config,run_training
    data = Path("/tmp/deepfake_data")
    data.mkdir(exist_ok=True)
    if not (data/".extraction_complete").exists():
        with tarfile.open("/mnt/deepfake/data/modal_faceforensics.tar") as archive:
            archive.extractall(data,filter="data")
        (data/".extraction_complete").touch()
    config = training_config(model,condition,"/root/protocol",data,"/mnt/deepfake/generalization_upgrade/training")
    try:
        return run_training(config)
    finally:
        volume.commit()


@app.function(timeout=14400,volumes={"/mnt/deepfake":volume})
def suite():
    import sys
    import time
    sys.path.insert(0,"/root/project")
    from src.experiments.common import save_json
    start = time.monotonic()
    jobs = [(model,"control") for model in ("hybrid","xception","freq_cnn")]
    jobs += [(model,f"lomo_{method}") for method in ("Deepfakes","Face2Face","FaceSwap","NeuralTextures")
             for model in ("hybrid","xception")]
    results = []
    for model,condition in jobs:
        if time.monotonic()-start > 3.5*3600:
            raise TimeoutError("Training budget reached; resume completed jobs without retraining")
        result = train_job.remote(model,condition)
        results.append({"model":model,"condition":condition,"checkpoint":result['checkpoint']})
        volume.reload()
        save_json("/mnt/deepfake/generalization_upgrade/training/suite_progress.json",results)
        volume.commit()
    return results


@app.local_entrypoint()
def main():
    print(suite.remote())
