"""Build an equal-budget material-repeat protocol and verified DINO/motion cache.

No target-based ranking or quality filtering. Old validation IDs stay fixed.
Frozen features and RGB-derived labels may be reused; encoded parameters are
always recomputed with this manifest's train-only statistics.
"""
import copy
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import torch
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, read_manifest, fit_stats, collate, model_inputs
from fieldbridge.local_model import build_model
from fieldbridge.runtime import sha256
from engine_fixture import atomic, wait_if_paused
from prepare_observed_motion import prepare

ROOT=Path(__file__).resolve().parents[1]
MANIFEST='data/manifests/d3_balanced_repeat_512_128.jsonl'
STATS='data/manifests/d3_balanced_repeat_512_128.stats.json'
CACHE='data/prepared/d3_balanced_repeat_dinov3'
MOTION='data/prepared/d3_balanced_repeat_motion'


def select_repeat_protocol(rows,candidates,old_rows,seed=31):
    rng=random.Random(seed)
    training={r['sample_id']:r for r in rows if r['split']=='train'}
    grouped={}
    for group in candidates['groups']:
        if group['split']!='train':continue
        members=list(group['sample_ids']);rng.shuffle(members)
        if len(members)<2 or any(s not in training for s in members):raise ValueError('Invalid train pair candidate')
        for s in members:
            row=training[s]
            if (row['object_id'],row['action'],row['law_id'])!=(group['object_id'],group['action'],group['law_id']):raise ValueError('Candidate metadata identity mismatch')
        grouped.setdefault((group['law_id'],group['action']),[]).append((group,members[:2]))
    original=[r for r in old_rows if r['split']=='train']
    if len(original)!=512:raise ValueError('Reference must have 512 training samples')
    quotas=Counter((r['law_id'],r['action']) for r in original)
    all_objects={r['object_id'] for r in original}
    if all_objects!={r['object_id'] for r in training.values()}:raise ValueError('Reference object coverage differs')
    # No repeats exist for metal/plasticine with the exported hardening/softening
    # held fixed. Keep those original records; never relax this nuisance control.
    singles=[r['sample_id'] for r in original if (r['law_id'],r['action']) not in grouped]
    covered={training[s]['object_id'] for s in singles}
    for obj in sorted(all_objects-covered):
        singles.append(next(r['sample_id'] for r in original if r['object_id']==obj))
    used=set(singles);pairs=[];chosen=list(singles)
    for key,budget in sorted(quotas.items()):
        remaining=budget-sum((training[s]['law_id'],training[s]['action'])==key for s in singles)
        by_object=defaultdict(list)
        for group,members in grouped.get(key,[]):by_object[group['object_id']].append((group,members))
        objects=sorted(by_object);rng.shuffle(objects)
        for value in by_object.values():rng.shuffle(value)
        while remaining>=2:
            changed=False
            for obj in objects:
                if remaining<2:break
                while by_object[obj]:
                    group,members=by_object[obj].pop()
                    if used.intersection(members):continue
                    chosen.extend(members);used.update(members);remaining-=2;changed=True
                    pairs.append(dict(object_id=obj,action=group['action'],law_id=group['law_id'],sample_ids=members,
                        physics_metadata_sha256=group['physics_metadata_sha256'],render_metadata_sha256=group['render_metadata_sha256']))
                    break
            if not changed:break
        for row in original:
            if remaining and (row['law_id'],row['action'])==key and row['sample_id'] not in used:
                chosen.append(row['sample_id']);singles.append(row['sample_id']);used.add(row['sample_id']);remaining-=1
        if remaining:raise ValueError('Cannot fill matched law/action budget')
    if len(set(chosen))!=len(chosen):raise ValueError('Pair member reused')
    train=[copy.deepcopy(training[s]) for s in chosen]
    val=[copy.deepcopy(r) for r in old_rows if r['split']=='val']
    if len(train)!=512 or len(val)!=128 or {r['object_id'] for r in train}!=all_objects:raise ValueError('Protocol budget/object coverage mismatch')
    if Counter((r['law_id'],r['action']) for r in train)!=quotas:raise ValueError('Law/action distribution mismatch')
    if any(r['split']=='test' for r in train+val):raise ValueError('Test excluded')
    return sorted(train+val,key=lambda r:(r['split'],r['sample_id'])),pairs,singles


def main():
    os.chdir(ROOT);torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    out=ROOT/'runs/d3_balanced_prepare';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',phase='selecting_protocol',completed=0,total=640,reused_features=0,new_features=0,reused_motion=0,
        detail='保持原三类材料×动作配额、88物体与512/128预算；仅在严格可配对的材料内增加重复，验证ID不变。',
        test_untouched=True,scope='Material variants change several native parameters jointly. Metadata matching is not SI certification or an E-ranking label.')
    started=time.time()
    def update(**kwargs):
        report.update(kwargs,updated_unix=time.time(),elapsed_seconds=time.time()-started)
        atomic(ROOT/'docs/D3_SINGLE_VIDEO_PREPARE.json',report)
    try:
        source=ROOT/'data/manifests/material_pair_candidates.json';candidates=json.loads(source.read_text())
        audit=json.loads((ROOT/'docs/MATERIAL_PAIR_AUDIT.json').read_text())
        if audit['status']!='complete' or audit['candidate_index_sha256']!=sha256(source):raise ValueError('Candidate audit hash mismatch')
        full=read_manifest('data/manifests/deformfield_v1.jsonl');old=read_manifest('data/manifests/g1_diagnostic_512_128.jsonl')
        rows,pairs,singles=select_repeat_protocol(full,candidates,old)
        manifest=ROOT/MANIFEST;manifest.parent.mkdir(parents=True,exist_ok=True)
        text=''.join(json.dumps(r)+'\n' for r in rows)
        if manifest.exists() and manifest.read_text()!=text:raise ValueError('Existing D3 protocol differs')
        manifest.write_text(text);fit_stats(manifest,ROOT/STATS)
        atomic(ROOT/'data/manifests/d3_balanced_repeat_pairs.json',dict(seed=31,pairs=pairs,coverage_singles=singles,
            source_index_sha256=sha256(source),validation_ids_unchanged=True,ranking_supervision=False))
        law_counts=Counter(str(r['law_id']) for r in rows if r['split']=='train')
        update(paired_train_samples=2*len(pairs),coverage_train_samples=len(singles),law_counts=dict(law_counts),
            admission='law_action_counts_and_object_coverage_matched',
            rejected_draft='The initial 510 jelly / 1 metal / 1 plasticine draft failed coverage admission and was never trained.')
        cfg=load_config('configs/g1f_cache.json');cfg.data.manifest=MANIFEST;cfg.data.stats=STATS
        cfg.data.prepared_dir='';cfg.data.motion_targets_dir='';cfg.data.view_dropout=0.
        cfg.model.field_mode='legacy_visual';cfg.model.architecture='motion_temporal'
        raw={s:MaterialDataset(cfg.data,s,cfg.model.field_size,'legacy_visual') for s in ['train','val']}
        old_dir=ROOT/'data/prepared/g1f_dinov3l16_local';old_index=json.loads((old_dir/'index.json').read_text())
        old_motion=ROOT/'data/prepared/m2_observed_motion';mi=json.loads((old_motion/'index.json').read_text())
        if old_index['backbone_checkpoint_sha256']!=sha256(cfg.model.weights) or old_index['encoder_source_sha256']!=sha256(ROOT/'src/fieldbridge/model.py'):
            raise ValueError('Frozen encoder provenance mismatch')
        directory=ROOT/CACHE;directory.mkdir(parents=True,exist_ok=True);motion=ROOT/MOTION;motion.mkdir(parents=True,exist_ok=True)
        feature_index={k:v for k,v in old_index.items() if k not in ['files','sha256','file_hashes']}
        feature_index.update(manifest_sha256=sha256(manifest),files={},parent_index_sha256=sha256(old_dir/'index.json'))
        motion_index={k:v for k,v in mi.items() if k not in ['files','sha256']}
        motion_index.update(manifest_sha256=sha256(manifest),files={},sha256={},parent_index_sha256=sha256(old_motion/'index.json'))
        feature_hashes={};model=build_model(cfg.model).cuda().eval();encoder=model.frame_encoder
        parity=[];distribution=[]
        for split in ['train','val']:
            ds=raw[split]
            for i,row in enumerate(ds.rows):
                wait_if_paused(out);sid=row['sample_id'];name=hashlib.sha256(sid.encode()).hexdigest()+'.pt'
                reused=sid in old_index['files']
                if reused:
                    sample=torch.load(old_dir/old_index['files'][sid],map_location='cpu',weights_only=True)
                    if sample['sample_id']!=sid:raise ValueError('Reused feature ID mismatch')
                    sample['target'],sample['target_mask']=ds.codec.encode(row['parameters'])
                    sample['prior'],sample['prior_mask']=ds.codec.encode(row.get('prior_parameters',{}))
                    report['reused_features']+=1
                else:
                    sample=ds[i];rgb=sample['rgb'];shape=rgb.shape
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        features=encoder(rgb.flatten(0,1).cuda()).cpu().reshape(*shape[:2],encoder.dim,shape[-2]//encoder.patch_size,shape[-1]//encoder.patch_size)
                    sample['features']=features;sample['rgb']=torch.zeros((*shape[:3],1,1),dtype=torch.float32)
                    sample['field_mask']=sample['field_mask'].bool();report['new_features']+=1
                if not torch.isfinite(sample['features']).all():raise ValueError('Non-finite DINO features')
                tmp=directory/(name+'.tmp');torch.save(sample,tmp);tmp.replace(directory/name)
                feature_index['files'][sid]=name;feature_hashes[sid]=sha256(directory/name)
                if sid in mi['files']:
                    path=old_motion/mi['files'][sid]
                    if sha256(path)!=mi['sha256'][sid]:raise ValueError('Source motion file hash mismatch')
                    payload=path.read_bytes();tmp=motion/(name+'.tmp');tmp.write_bytes(payload);tmp.replace(motion/name)
                    report['reused_motion']+=1
                else:
                    distribution.append(prepare(row,cfg.data.frames,cfg.data.max_views,cfg.model.field_size,motion))
                motion_index['files'][sid]=name;motion_index['sha256'][sid]=sha256(motion/name)
                if i<2:
                    original=ds[i]
                    for key in ['target','target_mask','valid','times','fields','field_mask']:
                        if not torch.equal(original[key],sample[key]):raise ValueError('Reused cache label/time/mask mismatch: '+key)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        inputs=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in model_inputs(collate([s])).items()} for s in [original,sample]]
                        a,b=[model(**x)['mean'].float() for x in inputs]
                    delta=float((a-b).abs().max());parity.append(delta)
                    if delta>.001:raise ValueError('D3 feature parity failed')
                update(phase='building_verified_cache',completed=report['completed']+1)
        feature_index['file_sha256']=feature_hashes
        atomic(directory/'index.json',feature_index);atomic(motion/'index.json',motion_index)
        update(status='complete',phase='ready_for_single_video_controls',paired_train_samples=2*len(pairs),coverage_train_samples=len(singles),
            train_samples=512,val_samples=128,train_objects=88,val_objects=10,validation_ids_unchanged=True,
            manifest=MANIFEST,manifest_sha256=sha256(manifest),stats=STATS,cache=CACHE,motion_targets=MOTION,
            feature_index_sha256=sha256(directory/'index.json'),motion_index_sha256=sha256(motion/'index.json'),
            parity_samples=len(parity),parity_max_z_delta=max(parity),stats_fit_split='train',
            label_reencoding='All target/prior z values recomputed for the current train-only codec, including reused feature records.',
            comparison_note='Different train-only codecs: cross-sampling comparison must use raw errors or one explicit evaluation codec, not the two training z scales.')
        atomic(out/'status.json',dict(status='complete',reason='metadata_repeat_protocol_and_verified_cache_ready'))
    except Exception as error:
        update(status='failed',error=str(error));atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
