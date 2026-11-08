"""Inventory metadata-matched material variants without opening test records/packs.

Equal exported metadata is a candidate control, not certified SI equivalence or
a single-parameter stiffness intervention. Several material labels change jointly.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from fieldbridge.data import read_manifest
from engine_fixture import atomic, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
TARGET_KEYS={'E','nu','density','rho','yield_stress','sigma_y'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    os.chdir(ROOT);out=ROOT/'runs/material_pair_audit';out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in read_manifest('data/manifests/deformfield_v1.jsonl') if r['split'] in ['train','val']]
    groups=defaultdict(list);render_groups=defaultdict(list);counts={s:Counter() for s in ['train','val']}
    report=dict(status='running',completed=0,total=len(rows),splits={},missing_metadata=[],test_untouched=True,
        detail='只读完整训练/验证元数据：固定边界、动作、非目标数值设置和已导出的视角，寻找材料变化对照。',
        scope='Manifest structure validated; test records excluded before metadata analysis. No test pack or model predictions used. Exported metadata matching does not certify geometry scale, force units, hidden simulator settings or scalar softness order.')
    try:
        for row in rows:
            wait_if_paused(out);base=Path(row['pack']).parent;split=row['split'];c=counts[split];c['samples']+=1
            files=[base/'meta'/name for name in ['run_parameters.json','external_actions.json','boundary_conditions.json']]
            if not all(f.is_file() for f in files):
                report['missing_metadata'].append(row['sample_id']);report['completed']+=1;continue
            run,actions,boundary=[json.loads(f.read_text()) for f in files]
            nuisance={k:v for k,v in run['material_params'].items() if k not in TARGET_KEYS}
            protocol=dict(action=actions,boundary=boundary,material_nontarget=nuisance,time=run['time_params'],
                substep_dt=run['substep_dt'],frame_dt=run['frame_dt'],frame_num_simulated=run['frame_num_simulated'])
            key=(split,row['object_id'],row['action'],row['law_id'],digest(protocol))
            groups[key].append(row['sample_id'])
            render=dict(view_names=run['view_names'],render_frame_indices=run['render_frame_indices'],
                camera_distance_scale=run.get('camera_distance_scale'),sample_pack_resize=run.get('arch4_lmdb_resize'))
            render_groups[(*key,digest(render))].append(row['sample_id'])
            driven=actions.get('velocity_driven_boundary_conditions',[])
            c['velocity_driven_samples']+=bool(driven)
            c['nonzero_gravity_samples']+=actions.get('gravity',{}).get('magnitude',0)!=0
            c['explicit_force_history_key_present']+=any(k in actions for k in ['force_history','external_forces','force_driven_boundary_conditions'])
            report['completed']+=1
            if report['completed']%256==0:
                report['updated_unix']=time.time();atomic(ROOT/'docs/MATERIAL_PAIR_AUDIT.json',report)
        for split in ['train','val']:
            g={k:v for k,v in groups.items() if k[0]==split};rg={k:v for k,v in render_groups.items() if k[0]==split}
            candidates={k:v for k,v in rg.items() if len(v)>=2}
            report['splits'][split]=dict(counts[split],objects=len({r['object_id'] for r in rows if r['split']==split}),
                physics_metadata_groups=len(g),physics_groups_with_repeats=sum(len(v)>1 for v in g.values()),
                render_metadata_groups=len(rg),render_groups_with_repeats=len(candidates),
                objects_with_candidate_pairs=len({k[1] for k in candidates}),
                candidate_disjoint_pairs=sum(len(v)//2 for v in candidates.values()))
        candidate=dict(status='candidate_only_not_admitted_for_training',groups=[dict(split=k[0],object_id=k[1],action=k[2],law_id=k[3],
            physics_metadata_sha256=k[4],render_metadata_sha256=k[5],sample_ids=v) for k,v in render_groups.items() if len(v)>=2],
            limitations=['Multiple material parameters change jointly. No unconditional E ranking loss.',
                'No certified SI scale, measured reaction forces or complete camera calibration.',
                'Native engine numerical settings are matched, not transferable parameter equivalence.'])
        atomic(ROOT/'data/manifests/material_pair_candidates.json',candidate)
        report.update(status='complete',updated_unix=time.time(),candidate_index='data/manifests/material_pair_candidates.json',
            candidate_index_sha256=hashlib.sha256((ROOT/'data/manifests/material_pair_candidates.json').read_bytes()).hexdigest(),native_parameter_ranking_admitted=False)
        atomic(ROOT/'docs/MATERIAL_PAIR_AUDIT.json',report)
        atomic(out/'status.json',dict(status='complete',reason='metadata_only_material_variant_inventory_complete'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/MATERIAL_PAIR_AUDIT.json',report)
        atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
