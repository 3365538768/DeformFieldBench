import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_single_video_protocol import select_repeat_protocol


def test_repeat_selection_keeps_budget_objects_validation_and_disjoint_members():
    rows=[];groups=[]
    for i in range(260):
        obj=f'object_{i%87:03d}';ids=[]
        for j in range(3):
            identifier=f'train_{i}_{j}';ids.append(identifier)
            rows.append(dict(sample_id=identifier,object_id=obj,action='press',law_id=i%3,split='train'))
        groups.append(dict(split='train',object_id=obj,action='press',law_id=i%3,sample_ids=ids,
            physics_metadata_sha256=str(i),render_metadata_sha256='render'))
    rows.append(dict(sample_id='unpaired',object_id='unpaired',action='press',law_id=0,split='train'))
    val=[dict(sample_id=f'val_{i}',object_id='val',action='press',law_id=0,split='val') for i in range(128)]
    rows+=val;original=copy.deepcopy(rows)
    reference=rows[:511]+[rows[-129]]+val
    result,pairs,singles=select_repeat_protocol(rows,dict(groups=groups),reference)
    assert len(pairs)>100 and 2*len(pairs)+len(singles)==512
    train=[r for r in result if r['split']=='train']
    assert len(train)==len({r['sample_id'] for r in train})==512
    assert {r['object_id'] for r in train}=={r['object_id'] for r in rows if r['split']=='train'}
    assert [r for r in result if r['split']=='val']==sorted(val,key=lambda r:r['sample_id'])
    assert result==select_repeat_protocol(rows,dict(groups=groups),reference)[0]
    assert rows==original


def test_repeat_selection_preserves_missing_law_controls_and_law_action_counts():
    from collections import Counter
    rows=[];groups=[]
    for law in range(3):
        for i in range(180):
            rows.append(dict(sample_id=f'{law}_{i}',object_id=f'o{i%10}',action='press' if i%2 else 'bend',law_id=law,split='train'))
    for i in range(0,180,20):
        for j in range(10):
            ids=[f'0_{i+j}',f'0_{i+j+10}']
            groups.append(dict(split='train',object_id=f'o{j}',action='press' if j%2 else 'bend',law_id=0,
                sample_ids=ids,physics_metadata_sha256=str(i+j),render_metadata_sha256='r'))
    original=rows[:164]+rows[180:354]+rows[360:534]
    val=[dict(sample_id=f'v{i}',object_id='v',action='press',law_id=0,split='val') for i in range(128)]
    result,pairs,singles=select_repeat_protocol(rows+val,dict(groups=groups),original+val)
    train=[r for r in result if r['split']=='train']
    assert Counter((r['law_id'],r['action']) for r in train)==Counter((r['law_id'],r['action']) for r in original)
    assert {r['sample_id'] for r in train if r['law_id']!=0}=={r['sample_id'] for r in original if r['law_id']!=0}
    assert len(pairs)>50 and all(p['law_id']==0 for p in pairs)
