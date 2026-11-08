import sys
from pathlib import Path
import pytest
import numpy as np

cv2=pytest.importorskip('cv2')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from visible_motion import estimate


def test_translation_units_and_masked_endpoint():
    cv2.setNumThreads(1)
    rng=np.random.default_rng(17)
    a=rng.integers(0,256,size=(96,96,3),dtype=np.uint8)
    b=np.roll(a,3,axis=1)
    mask=np.zeros((96,96),bool);mask[16:80,16:70]=True
    endpoint=np.roll(mask,3,axis=1)
    flow,valid,_=estimate(a,b,mask,endpoint)
    assert valid.sum()>.9*mask.sum()
    assert np.median(np.linalg.norm(flow[valid]-[3.,0.],axis=1))<.2
    assert not valid[~mask].any()
    _,empty,_=estimate(a,b,mask,np.zeros_like(mask))
    assert not empty.any()
