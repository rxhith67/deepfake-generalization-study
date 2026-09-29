"""CPU-only interrupted-run equivalence, without downloading a backbone."""
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.config import load_config
import src.train as trainer


def test_resume_restores_optimizer_scheduler_and_rng(tmp_path,monkeypatch):
    def tiny_model(*args,**kwargs):
        return nn.Sequential(nn.Flatten(),nn.Linear(3*8*8,1))
    monkeypatch.setattr(trainer,'build_model',tiny_model)
    rows = []
    for i in range(4):
        path = tmp_path/f'{i}.png'
        cv2.imwrite(str(path),np.full((8,8,3),i*50,dtype=np.uint8))
        rows.append({'image_path':str(path),'label':i%2})
    manifest = tmp_path/'samples.csv'
    pd.DataFrame(rows).to_csv(manifest,index=False)
    config = load_config('config/meso.yaml')
    config['device'] = 'cpu'
    config['data'].update(root=str(tmp_path),num_workers=0,train_csv=str(manifest),val_csv=str(manifest),test_csv=str(manifest))
    config['training'].update(image_size=8,epochs=2,batch_size=2,resumable=True,amp=False,early_stopping_patience=3)
    config['augmentation'] = {'hflip':0,'rotate_limit':0,'color_jitter':0,'jpeg_prob':0}
    config['logging']['output_dir'] = str(tmp_path/'interrupted')
    original = trainer.train_one_epoch
    calls = []
    def interrupt(*args,**kwargs):
        if calls:
            raise RuntimeError('simulated interruption')
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(trainer,'train_one_epoch',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        trainer.train(config)
    monkeypatch.setattr(trainer,'train_one_epoch',original)
    trainer.train(config)
    resumed = torch.load(tmp_path/'interrupted/checkpoints/training_state_last.pt',weights_only=False)
    config['logging']['output_dir'] = str(tmp_path/'uninterrupted')
    trainer.train(config)
    fresh = torch.load(tmp_path/'uninterrupted/checkpoints/training_state_last.pt',weights_only=False)
    assert resumed['next_epoch'] == fresh['next_epoch'] == 2
    assert resumed['scheduler'] == fresh['scheduler']
    for name,value in resumed['model_state'].items():
        torch.testing.assert_close(value,fresh['model_state'][name],rtol=0,atol=0)
