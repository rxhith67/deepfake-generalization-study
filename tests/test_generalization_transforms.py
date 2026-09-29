import numpy as np
import pandas as pd
import pytest
from src.experiments.corruptions import jpeg,filter_signal,frequency_filter
from src.experiments.external import align_face,TEMPLATE
from src.experiments.evaluation import strict_ensemble
from src.experiments.calibration import select_threshold


def test_jpeg_even_q100_is_encoded_and_deterministic():
    image = np.random.default_rng(42).integers(0,256,(224,224,3),dtype=np.uint8)
    out = jpeg(image,100)
    assert out.dtype == np.uint8 and out.shape == image.shape
    assert not np.array_equal(image,out)
    assert np.array_equal(out,jpeg(image,100))


def test_frequency_complement_and_constant():
    image = np.ones((32,32))*.7
    assert np.allclose(filter_signal(image,'lowpass',.3),image)
    assert np.allclose(filter_signal(image,'highpass',.3),0)
    signal = np.random.default_rng(42).normal(size=(32,32,3))
    assert np.allclose(filter_signal(signal,'lowpass',.3)+filter_signal(signal,'highpass',.3),signal)
    with pytest.raises(ValueError):
        filter_signal(signal,'invalid',.3)


def test_lowpass_retains_low_sinusoid_removes_high():
    x = np.arange(100)
    lo = np.tile(np.cos(2*np.pi*.02*x),(100,1))
    hi = np.tile(np.cos(2*np.pi*.4*x),(100,1))
    assert np.allclose(filter_signal(lo+hi,'lowpass',.1),lo,atol=1e-10)


def test_alignment_identity_and_shape():
    image = np.random.default_rng(1).integers(0,256,(224,224,3),dtype=np.uint8)
    out,matrix = align_face(image,TEMPLATE*2)
    assert np.allclose(matrix,np.array([[1,0,0],[0,1,0]]),atol=1e-5)
    assert np.array_equal(out,image)
    with pytest.raises(ValueError):
        align_face(image,np.ones((3,2)))


def test_ensemble_exact_coverage_and_threshold_validation_only():
    frame = pd.DataFrame({'sample_id':['a','b'],'label':[0,1],'prob_fake':[.2,.8],
                          'split':['val']*2,'dataset':['ffpp_control']*2})
    assert 0 <= select_threshold(frame) <= 1
    with pytest.raises(ValueError):
        select_threshold(frame.assign(split='external_test'))
    with pytest.raises(ValueError):
        strict_ensemble([frame,frame.iloc[:1]],[.5,.5],.5)
    out = strict_ensemble([frame,frame],[.5,.5],.6)
    assert list(out.prediction) == [0,1]
