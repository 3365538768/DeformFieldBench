from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from interaction_protocol import force_scale,select_motion_frames,should_stop


def test_motion_sampling_removes_static_tail_without_retiming():
    trace=np.zeros((200,4,3));trace[:10,:,1]=np.linspace(.001,.01,10)[:,None];trace[10:,:,1]=.01
    indices,receipt=select_motion_frames(trace,.02)
    assert indices[0]==0 and indices[-1]<=13
    assert receipt['dropped_static_tail_frames']>=187
    assert receipt['selected_active_transition_fraction']>receipt['source_active_transition_fraction']*10
    assert len(indices)<=17 and np.all(np.diff(indices)>0)
    assert receipt['retained_duration_s']==pytest.approx(indices[-1]*.02)


def test_unloading_retains_peak_and_future_action_prevents_early_stop():
    trace=np.zeros((100,4,3));trace[:10,:,1]=np.linspace(0,.01,10)[:,None]
    trace[10:20,:,1]=np.linspace(.01,0,10)[:,None]
    indices,_=select_motion_frames(trace,.02)
    assert 10 in indices and indices[-1]>=20
    case=dict(stop_criteria={},force_history=[[0,0],[2,1],[3,0]])
    case['stop_criteria']=dict(minimum_steps=20)
    assert not should_stop(list(np.zeros((40,4,3))),0,.8,case)
    assert should_stop(list(trace[:40]),0,3.1,case)
    assert force_scale(.5,[[0,0],[1,1],[2,0]])==pytest.approx(.5)
    with pytest.raises(ValueError):force_scale(1,[[1,0],[0,1]])
