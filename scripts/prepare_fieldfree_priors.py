"""First-frame ridge priors with object-out-of-fold training predictions."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from fieldbridge.config import load_config
from fieldbridge.data import read_manifest, ParameterCodec
from fieldbridge.runtime import sha256
from engine_fixture import atomic, wait_if_paused

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'data/prepared/nf_static_priors.json'


def fit_ridge(x, y, mask, seed=17):
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(x.shape[-1], 256, generator=generator)/(x.shape[-1]**.5)
    center = x.mean(0); scale = x.std(0,unbiased=False).clamp_min(.05)
    def design(value):
        z = ((value-center)/scale).clamp(-10,10)@projection
        return torch.cat((z,torch.ones(len(z),1)),1)
    a = design(x); weights=[]; variances=[]
    for j in range(4):
        keep=mask[:,j]
        if not keep.any():
            weights.append(torch.zeros(a.shape[1]));variances.append(torch.tensor(4.));continue
        xx=a[keep];yy=y[keep,j]
        penalty=torch.eye(a.shape[1]);penalty[-1,-1]=1e-6
        w=torch.linalg.solve(xx.T@xx+penalty,xx.T@yy)
        residual=(xx@w-yy).square().mean()
        weights.append(w);variances.append(residual.clamp(.1,4.))
    return dict(projection=projection,center=center,scale=scale,weights=torch.stack(weights,1),
                logvar=torch.stack(variances).log(),mask=mask.any(0))


def ridge_predict(fit, x):
    z=((x-fit['center'])/fit['scale']).clamp(-10,10)@fit['projection']
    return torch.cat((z,torch.ones(len(z),1)),1)@fit['weights']


def object_folds(rows, count=5):
    objects=sorted({r['object_id'] for r in rows if r['split']=='train'})
    objects.sort(key=lambda name:hashlib.sha256(('nf17:'+name).encode()).hexdigest())
    return {name:i%count for i,name in enumerate(objects)}


def main():
    os.chdir(ROOT);torch.set_num_threads(1)
    cfg=load_config('configs/single_video_base.json');rows=[r for r in read_manifest(cfg.data.manifest) if r['split'] in ['train','val']]
    base=ROOT/cfg.data.prepared_dir;index=json.loads((base/'index.json').read_text());codec=ParameterCodec(json.loads(Path(cfg.data.stats).read_text()))
    out=ROOT/'runs/nf_static_prepare';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=len(rows),detail='首个有效帧DINO特征的物体级五折静态先验；训练OOF、验证仅全训练物体拟合，不读场标签。',test_untouched=True)
    def update(**values):
        report.update(values,updated_unix=time.time());atomic(ROOT/'docs/NF_STATIC_PREPARE.json',report)
    try:
        records=[]
        for row in rows:
            wait_if_paused(out);p=torch.load(base/index['files'][row['sample_id']],weights_only=True)
            if p['sample_id']!=row['sample_id']:raise ValueError('Static input identity mismatch')
            vectors=[]
            for v in range(p['valid'].shape[0]):
                ts=torch.where(p['valid'][v])[0]
                vectors.append(p['features'][v,int(ts[0])].float().mean((-1,-2)) if len(ts) else torch.zeros(p['features'].shape[2]))
            records.append(dict(row=row,x=torch.stack(vectors),valid=p['valid'].any(1),
                                input_sha256=sha256(base/index['files'][row['sample_id']])))
            update(completed=len(records))
        fold=object_folds(rows);fits={};fitted={}
        for fit_id in ['fold0','fold1','fold2','fold3','fold4','full_train']:
            wait_if_paused(out)
            selected=[r for r in records if r['row']['split']=='train' and (fit_id=='full_train' or fold[r['row']['object_id']]!=int(fit_id[-1]))]
            x=torch.cat([r['x'][r['valid']] for r in selected]);targets=[];masks=[]
            for r in selected:
                y,m=codec.encode(r['row']['parameters']);n=int(r['valid'].sum());targets.append(y.repeat(n,1));masks.append(m.repeat(n,1))
            fit=fit_ridge(x,torch.cat(targets),torch.cat(masks));path=out/(fit_id+'.pt');torch.save(fit,path);fitted[fit_id]=fit
            fits[fit_id]=dict(training_objects=sorted({r['row']['object_id'] for r in selected}),
                             checkpoint_sha256=sha256(path),checkpoint=str(path.relative_to(ROOT)),
                             protocol='train_object_out_of_fold' if fit_id!='full_train' else 'train_split_only')
            update(fitted_models=len(fits),fit_total=6)
        result=dict(schema='fieldfree_static_prior_v1',manifest_sha256=sha256(cfg.data.manifest),stats_sha256=sha256(cfg.data.stats),
                    feature_index_sha256=sha256(base/'index.json'),fits=fits,samples={},
                    scope='Only first valid frame per source camera; fixed ridge recipe, no val tuning, no pretrained semantic material model.')
        val_errors=[]
        for r in records:
            row=r['row'];fit_id='full_train' if row['split']=='val' else 'fold'+str(fold[row['object_id']]);fit=fitted[fit_id]
            mean=ridge_predict(fit,r['x']);v=len(mean)
            result['samples'][row['sample_id']]=dict(object_id=row['object_id'],fit_id=fit_id,mean_z=mean.tolist(),
                logvar_z=fit['logvar'].repeat(v,1).tolist(),mask=(fit['mask'][None]&r['valid'][:,None]).tolist(),input_sha256=r['input_sha256'])
            if row['split']=='val':
                y,m=codec.encode(row['parameters']);first=int(torch.where(r['valid'])[0][0]);val_errors.extend((mean[first]-y).abs()[m].tolist())
        atomic(ROOT/OUTPUT,result)
        update(status='complete',prior_file=OUTPUT,prior_sha256=sha256(ROOT/OUTPUT),fits=6,
               static_validation_mae=sum(val_errors)/len(val_errors),training_protocol='five object folds; no sample sees own-object fitted labels',
               uncertainty='Training-fit residual variance, uncalibrated; not a verified physical posterior.',
               limitation='Existing synthetic appearances may be decoupled from assigned materials; this is a static prior baseline, not PIXIE/VoMP integration.')
        atomic(out/'status.json',dict(status='complete',reason='object_oof_static_prior_ready'))
    except Exception as error:
        update(status='failed',error=str(error));atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
