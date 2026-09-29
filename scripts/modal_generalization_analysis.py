"""Wait for the approved 11 training jobs, then run Tier 1 analysis and Tier 2.

No Tier 3/4 tasks. CPU waiting has a four-hour limit; one A10 analysis job.
python -X utf8 -m modal run --detach scripts/modal_generalization_analysis.py
"""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("numpy==1.26.4","torch==2.2.2","torchvision==0.17.2","timm==1.0.7",
                         "opencv-python-headless==4.10.0.84","albumentations==1.3.1","scikit-learn==1.5.1",
                         "pandas==2.2.2","pyyaml==6.0.2","tqdm==4.66.5","huggingface-hub==0.34.4",
                         "matplotlib==3.9.2","grad-cam==1.5.4","diffusers==0.30.3","lpips==0.1.4",
                         "scikit-image==0.24.0","accelerate==0.34.2")
         .env({"HF_HOME":"/mnt/deepfake/generalization_upgrade/cache/huggingface",
               "TORCH_HOME":"/mnt/deepfake/generalization_upgrade/cache/torch"})
         .add_local_dir(ROOT/'src',remote_path='/root/project/src',copy=True)
         .add_local_dir(ROOT/'config',remote_path='/root/project/config',copy=True)
         .add_local_dir(ROOT/'outputs/generalization_upgrade/protocol',remote_path='/root/protocol',copy=True)
         .add_local_dir(ROOT/'outputs/generalization_upgrade/cloud_inputs',remote_path='/root/external',copy=True))
app = modal.App('deepfake-generalization-analysis',image=image)
volume = modal.Volume.from_name('deepfake-detection')


@app.function(gpu='A10',cpu=8,memory=32768,timeout=7200,max_containers=1,volumes={'/mnt/deepfake':volume})
def analyze():
    import os
    import sys
    import tarfile
    sys.path.insert(0,'/root/project');os.chdir('/root/project')
    from src.experiments.common import configuration,save_json
    from src.experiments.study import run
    from src.experiments.reconstruction import run as reconstruct
    data = Path('/tmp/deepfake_data');data.mkdir(exist_ok=True)
    for path in ['/mnt/deepfake/data/modal_faceforensics.tar','/root/external/external.tar']:
        with tarfile.open(path) as archive:
            archive.extractall(data,filter='data')
    config = configuration()
    config.update(device='cuda',batch_size=32,output_dir='/mnt/deepfake/generalization_upgrade/analysis')
    config['data']['root'] = str(data)
    original,normalized = Path('/root/external/original.csv'),Path('/root/external/normalized.csv')
    try:
        run(config,'/mnt/deepfake/generalization_upgrade/training','/root/protocol',original,normalized)
        save_json(Path(config['output_dir'])/'tier1_analysis_complete.json',{'status':'complete'})
        volume.commit()
        reconstruct(config,Path('/root/protocol/control/test.csv'),normalized)
        save_json(Path(config['output_dir'])/'tier2_complete.json',{'status':'complete'})
        return {'tier1':'complete','tier2':'complete','output':config['output_dir']}
    finally:
        volume.commit()


@app.function(cpu=.25,memory=512,timeout=21600,volumes={'/mnt/deepfake':volume})
def after_training():
    import json
    import time
    progress = Path('/mnt/deepfake/generalization_upgrade/training/suite_progress.json')
    deadline = time.monotonic()+4*3600
    expected = {(m,'control') for m in ['hybrid','xception','freq_cnn']}
    expected |= {(m,f'lomo_{held}') for m in ['hybrid','xception'] for held in ['Deepfakes','Face2Face','FaceSwap','NeuralTextures']}
    while time.monotonic()<deadline:
        volume.reload()
        if progress.exists():
            jobs = json.loads(progress.read_text())
            if {(j['model'],j['condition']) for j in jobs} == expected:
                return analyze.remote()
        time.sleep(30)
    raise TimeoutError('Required training jobs did not finish in the bounded waiting period; no analysis GPU was started')


@app.local_entrypoint()
def main():
    print(after_training.remote())
