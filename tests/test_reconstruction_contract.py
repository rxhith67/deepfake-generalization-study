from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.experiments.reconstruction import fake_score,reconstruct
from src.eval.calibration import bootstrap_auc


def test_score_orientation_is_fixed_not_selected_from_labels():
    assert np.array_equal(fake_score('mse',[.1,.8]),[-.1,-.8])
    assert np.array_equal(fake_score('lpips',[.1,.8]),[-.1,-.8])
    assert np.array_equal(fake_score('ssim',[.1,.8]),[.1,.8])
    with pytest.raises(ValueError):
        fake_score('unknown',[1])
    assert bootstrap_auc([0,0,1,1],[-9,-8,-2,-1],replicates=50,probability_score=False)['auc_ci_lower']==1


def test_direct_vae_roundtrip_uses_mode_without_latent_rescaling():
    class Stub:
        def encode(self,x):
            assert x.min()>=-1 and x.max()<=1
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:x))
        def decode(self,z):
            return SimpleNamespace(sample=z)
    rgb = torch.rand(2,3,16,16)
    torch.testing.assert_close(reconstruct(Stub(),rgb),rgb)
    with pytest.raises(ValueError):
        reconstruct(Stub(),rgb*4)
