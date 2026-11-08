from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_single_video_controls import common_records
from fieldbridge.data import ParameterCodec


def test_common_codec_preserves_native_predictions_masks_and_pairing():
    source=dict(mean=[10.,.4,7.,5.],std=[2.,.05,1.,2.])
    reference=dict(mean=[9.,.3,6.,6.],std=[3.,.1,2.,1.])
    codec=ParameterCodec(source);target_codec=ParameterCodec(reference)
    params=dict(E=10000.,nu=.35,rho=900.,sigma_y=None)
    z,mask=codec.encode(params);records=[];rows={}
    for i in range(128):
        sid=str(i);rows[sid]=dict(object_id='o'+str(i%10),parameters=params)
        records.append(dict(sample_id=sid,object_id=rows[sid]['object_id'],target_z=z.tolist(),target_mask=mask.tolist(),prediction_z=(z+.2).tolist()))
    result=common_records(records,source,reference,rows)
    for record in result.values():
        expected=target_codec.encode(params)[0].numpy();expected[:3]+=.2*np.array(source['std'][:3])/reference['std'][:3]
        np.testing.assert_allclose(record['prediction_z'][:3],expected[:3],atol=1e-6)
        assert record['target_mask']==[True,True,True,False]
    records[0]['target_z'][0]+=1
    with pytest.raises(ValueError,match='Codec target'):common_records(records,source,reference,rows)
