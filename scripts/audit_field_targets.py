"""Train/validation target structure and fixed ridge probes; test never loaded."""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate
from fieldbridge.runtime import atomic_json, sha256


def describe(fields, mask, valid):
    # Seven numbers per channel: global mean/std, temporal delta, 2x2 regions.
    # Foreground support follows existing targets; no inferred SI interpretation.
    reduce=(0,1,3,4);count=mask.sum(reduce).clamp_min(1)
    mean=(fields*mask).sum(reduce)/count
    std=(((fields-mean[None,None,:,None,None])**2*mask).sum(reduce)/count).sqrt()
    pair=mask[:,1:]*mask[:,:-1]*(valid[:,1:]&valid[:,:-1])[...,None,None,None]
    delta=((fields[:,1:]-fields[:,:-1]).abs()*pair).sum(reduce)/pair.sum(reduce).clamp_min(1)
    spatial=(F.adaptive_avg_pool2d((fields*mask).flatten(0,1),2).sum(0)/
             F.adaptive_avg_pool2d(mask.flatten(0,1),2).sum(0).clamp_min(1e-8)).flatten()
    return torch.cat((mean,std,delta,spatial)),mean,std,delta


def ridge_probe(train,val,key,law_center=False):
    x=np.stack([r[key] for r in train]);z=np.stack([r[key] for r in val])
    mean=x.mean(0);std=np.maximum(x.std(0),.01)
    x=(x-mean)/std;z=(z-mean)/std
    if law_center:
        for law in sorted(set(r['law'] for r in train)):
            a=np.array([r['law']==law for r in train]);b=np.array([r['law']==law for r in val])
            center=x[a].mean(0);x[a]-=center;z[b]-=center
    errors=[];params={}
    for j,name in enumerate(['E','nu','rho','sigma_y']):
        a=np.array([r['target_mask'][j] for r in train],bool);b=np.array([r['target_mask'][j] for r in val],bool)
        if not a.any() or not b.any():continue
        y=np.array([r['target'][j] for r in train])[a];center=y.mean()
        train_center=np.full(len(train),center);val_center=np.full(len(val),center)
        if law_center:
            for law in sorted(set(r['law'] for r in train)):
                aa=np.array([r['law']==law for r in train])&a
                if aa.any():
                    c=float(np.mean([r['target'][j] for r in train if r['law']==law and r['target_mask'][j]]))
                    train_center[np.array([r['law']==law for r in train])]=c
                    val_center[np.array([r['law']==law for r in val])]=c
        weights=np.linalg.solve(x[a].T@x[a]+np.eye(x.shape[1]),x[a].T@(y-train_center[a]))
        pred=z[b]@weights+val_center[b];gt=np.array([r['target'][j] for r in val])[b]
        error=np.abs(pred-gt);errors.extend(error.tolist())
        params[name]={'n':int(b.sum()),'mae':float(error.mean()),'train_mean_mae':float(np.abs(gt-val_center[b]).mean())}
    return {'standardized_mae':float(np.mean(errors)),'parameters':params,'dimensions':int(x.shape[1]),
            'lambda':1.,'law_centered':law_center,'fit':'train only; fixed lambda; no hyperparameter selection'}


def run(config,output):
    cfg=load_config(config);torch.set_num_threads(1);rows={};summaries={};hashes=defaultdict(list)
    for split in ['train','val']:
        ds=MaterialDataset(cfg.data,split,cfg.model.field_size,cfg.model.field_mode)
        selected=[];sums=[];positive=contact_count=0;empty=0
        for batch in DataLoader(ds,batch_size=4,num_workers=2,collate_fn=collate):
            for i,identifier in enumerate(batch['sample_id']):
                f=batch['fields'][i].float();m=batch['field_mask'][i].float();valid=batch['valid'][i]
                descriptor,mean,std,delta=describe(f,m,valid)
                p=((f[:,:,:1]>.5)*m[:,:,:1]).sum().item();n=m[:,:,:1].sum().item()
                positive+=p;contact_count+=n;empty+=p==0
                rgb=batch['features'][i].float().mean((-1,-2))
                weights=valid[...,None];rgb_mean=(rgb*weights).sum((0,1))/weights.sum().clamp_min(1)
                rgb_delta=((rgb[:,1:]-rgb[:,:-1]).abs()*(valid[:,1:]&valid[:,:-1])[...,None]).sum((0,1))/(valid[:,1:]&valid[:,:-1]).sum().clamp_min(1)
                selected.append({'id':identifier,'object':batch['object_id'][i],
                    'field':descriptor.numpy().astype(np.float64),'rgb':torch.cat((rgb_mean,rgb_delta)).numpy().astype(np.float64),
                    'target':batch['target'][i].numpy(),'target_mask':batch['target_mask'][i].numpy(),
                    'law':int(batch['law'][i]),'mean':mean.tolist(),'std':std.tolist(),'delta':delta.tolist()})
                digest=hashlib.sha256(f.numpy().tobytes()).hexdigest();hashes[digest].append((split,identifier))
                sums.append(torch.stack((mean,std,delta)))
            atomic_json(str(output)+'.progress.json',{'phase':'target_audit','split':split,'processed':len(selected),'total':len(ds)})
        rows[split]=selected;a=torch.stack(sums)
        groups=defaultdict(list)
        for r in selected:groups[r['object']].append(r)
        pairs=different=0
        for group in groups.values():
            for i,r in enumerate(group):
                for q in group[i+1:]:
                    common=r['target_mask']&q['target_mask']
                    if common.any() and np.max(np.abs(r['target'][common]-q['target'][common]))>.1:
                        pairs+=1;different+=float(np.linalg.norm(r['field']-q['field']))>1e-6
        summaries[split]={'samples':len(ds),'objects':len(groups),'contact_positive_fraction':positive/max(contact_count,1),
                          'samples_without_contact':empty,'channel_mean':a[:,0].mean(0).tolist(),
                          'channel_supported_std':a[:,1].mean(0).tolist(),'channel_temporal_abs_delta':a[:,2].mean(0).tolist(),
                          'same_object_different_parameter_pairs':pairs,'pairs_with_distinct_field_descriptors':different,
                          'law_counts':dict(Counter(r['law'] for r in selected))}
    duplicate=[v for v in hashes.values() if len(v)>1]
    train,val=rows['train'],rows['val']
    rng=np.random.default_rng(17);permutation=rng.permutation(len(val))
    shuffled=[dict(r,field=val[int(permutation[i])]['field']) for i,r in enumerate(val)]
    groups=defaultdict(list)
    for i,r in enumerate(val):groups[(r['object'],r['law'])].append(i)
    donor=np.arange(len(val))
    for members in groups.values():
        if len(members)>1:donor[members]=np.roll(members,1)
    conditional=[dict(r,field=val[int(donor[i])]['field']) for i,r in enumerate(val)]
    result={'manifest_sha256':sha256(cfg.data.manifest),'splits':summaries,
            'probes':{'privileged_true_fields':ridge_probe(train,val,'field'),
                      'val_fields_permuted':ridge_probe(train,shuffled,'field'),
                      'val_fields_permuted_within_object_law':ridge_probe(train,conditional,'field'),
                      'privileged_true_fields_law_centered':ridge_probe(train,val,'field',True),
                      'val_fields_permuted_within_object_law_centered':ridge_probe(train,conditional,'field',True),
                      'frozen_rgb_features':ridge_probe(train,val,'rgb')},
            'within_object_law_permutation_changed_samples':int((donor!=np.arange(len(val))).sum()),
            'exact_field_duplicate_groups':len(duplicate),'cross_split_exact_duplicates':sum(len(set(s for s,_ in v))>1 for v in duplicate),
            'note':'Exploratory train-fit/val-only ridge probes with unequal descriptor capacity, NOT a method comparison, physical identifiability proof, matched actions or independent test. Same object does not imply identical appearance/load.'}
    atomic_json(output,result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)
