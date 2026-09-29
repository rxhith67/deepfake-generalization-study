import numpy as np
import pandas as pd
import pytest

from src.eval.calibration import calibration_metrics, reliability, bootstrap_auc
from src.experiments.manifests import canonical_id, derive, lomo, assert_disjoint


def frame(stems):
    return pd.DataFrame([{"image_path": f"faceforensics/{method}/{stem}/000000.jpg", "video_id": int(stem.split('_')[0]),
                          "method": method, "label": int(method != 'real')} for stem, method in stems])


def test_derive_preserves_targets_and_removes_donors_across_partitions():
    parts = {"train": frame([('001','real'), ('002','real'), ('001_002','Deepfakes'), ('001_003','FaceSwap')]),
             "val": frame([('003','real')]), "test": frame([('004','real')])}
    safe, excluded = derive(parts)
    assert len(safe['train']) == 3
    assert len(excluded) == 1
    assert_disjoint(safe)
    assert len(parts['train']) == 4


def test_canonical_ids_match_windows_and_cloud():
    assert canonical_id(r'C:\data\processed\faceforensics\real\001\1.jpg') == canonical_id('faceforensics/real/001/1.jpg')
    with pytest.raises(ValueError):
        canonical_id('ambiguous/1.jpg')


def test_lomo_excludes_heldout_from_train_and_validation():
    parts = {s: frame([(v,'real'), (v+'_'+v,'Deepfakes'), (v+'_'+v,'FaceSwap')])
             for s,v in [('train','001'),('val','002'),('test','003')]}
    safe, _ = derive(parts)
    result = lomo(safe, 'Deepfakes')
    assert 'Deepfakes' not in set(result['val'].method)
    assert set(result['test'].method) == {'real','Deepfakes'}


def test_calibration_perfect_and_uninformative():
    assert calibration_metrics([0,1], [0,1])['ece'] == 0
    result = calibration_metrics([0,1], [.5,.5])
    assert result['ece'] == 0
    assert result['brier'] == .25
    assert result['nll'] == pytest.approx(np.log(2))
    assert reliability([1],[1])[-1]['count'] == 1
    with pytest.raises(ValueError):
        calibration_metrics([0], [float('nan')])


def test_bootstrap_reproducible_clustered():
    args = ([0,0,1,1,0,1], [.1,.2,.8,.7,.4,.6])
    a = bootstrap_auc(*args, groups=['a','a','b','b','c','c'], replicates=50)
    assert a == bootstrap_auc(*args, groups=['a','a','b','b','c','c'], replicates=50)
    assert a['auc_ci_lower'] == 1
