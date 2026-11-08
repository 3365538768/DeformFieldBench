import importlib.util
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('motion_audit',Path(__file__).parents[1]/'scripts/audit_motion_dependence.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_interventions_preserve_padding_and_leave_targets_and_times_fixed():
    x=torch.arange(6).reshape(1,2,3,1,1,1)
    batch=dict(features=x,rgb=x.clone(),valid=torch.tensor([[[True,True,False],[False]*3]]),
               times=torch.arange(3),target=torch.ones(1,4))
    reversed=audit.intervene(batch,'reverse_observations')
    assert reversed['features'].flatten().tolist()==[1,0,2,3,4,5]
    assert torch.equal(batch['features'],x)
    assert reversed['times'] is batch['times'] and reversed['target'] is batch['target']
    static=audit.intervene(batch,'repeat_first_frame')
    assert static['features'].flatten().tolist()==[0,0,2,3,4,5]


def test_cluster_bootstrap_preserves_repeated_object_labels():
    left={str(i):dict(object_id=str(i//3),prediction_z=[.3],target_z=[0.],target_mask=[True]) for i in range(9)}
    right={k:dict(v,prediction_z=[.1]) for k,v in left.items()}
    result=audit.paired_bootstrap(left,right)
    assert result['objects']==3 and result['samples']==9
    assert abs(result['delta_mae_left_minus_right']-.2)<1e-10
    assert all(abs(x-.2)<1e-10 for x in result['exploratory_95_cluster_interval'])
