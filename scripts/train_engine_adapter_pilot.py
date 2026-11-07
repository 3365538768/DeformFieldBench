"""Observed-field pilot: common encoder + calibrated native engine heads.

No RGB, no density, no online-fitting claim. Validation selects checkpoints and
is NOT a final test. A canonical held-load response is supplied on TRAIN only.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from engine_fixture import atomic, wait_if_paused
from fieldbridge.engine_domains import EngineConditionedHeads, EngineDomain, require_matched_response


ROOT = Path(__file__).resolve().parents[1]


def arrays(manifest):
    rows = manifest['rows']; groups = {}
    for row in rows:
        groups.setdefault((row['fixture'],row['material_pair'],row['engine']),{})[row['axis']] = row
    samples = []
    for (fixture, material, engine), actions in sorted(groups.items()):
        if set(actions) != {0,1,2}:
            raise ValueError('Need observed X/Y and held response Z for each group')
        row = actions[0]; case = row['case']
        observed = [np.load(ROOT/actions[a]['path']/'field.npz')['displacement_m'].reshape(-1) for a in (0,1)]
        canonical = groups[(fixture,material,'warp')][2]
        require_matched_response(actions[2]['response_provenance'], canonical['response_provenance'])
        domain = dict(engine='NVIDIA Warp FEM' if engine=='warp' else 'libuipc', version=row['engine_version'],
            constitutive_law='small_strain_linear_elasticity' if engine=='warp' else 'SNK1',
            coefficient_mapping='Lame_mu_lambda' if engine=='warp' else 'explicit_muL_lambdaL_plus_muL',
            solver_protocol='g2_known_traction_block_v1',
            parameter_units={'mu':'Pa','lambda_lame' if engine=='warp' else 'lambda_snk1':'Pa'})
        names = ['mu','lambda_lame' if engine=='warp' else 'lambda_snk1']
        samples.append(dict(fixture=fixture,material=material,engine=engine,domain=domain,
            domain_id=EngineDomain.parse(domain).key,split=row['split'],parameters=names,
            target=[row['native_coefficients_pa'][n] for n in names],
            features=np.concatenate([np.array(case['dimensions_m']),*observed]),
            response=canonical['tip_displacement_m']))
    if set(s['fixture'] for s in samples if s['split']=='train') & set(s['fixture'] for s in samples if s['split']=='val'):
        raise ValueError('Fixture leakage')
    return samples


class Pilot(nn.Module):
    def __init__(self, inputs, specs):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(inputs,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.adapter=EngineConditionedHeads(64,specs,['held_Z_tip_x_m','held_Z_tip_y_m','held_Z_tip_z_m'])

    def forward(self,x,domains):
        latent=self.encoder(x); return latent,self.adapter(latent,domains)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); os.chdir(ROOT);torch.set_num_threads(1)
    manifest_path=Path(args.manifest);manifest=json.loads(manifest_path.read_text());samples=arrays(manifest)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    train=np.array([i for i,s in enumerate(samples) if s['split']=='train']);val=np.array([i for i,s in enumerate(samples) if s['split']=='val'])
    train_hash=hashlib.sha256(json.dumps([{k:v for k,v in s.items() if k not in {'features'}} for s in samples if s['split']=='train'],sort_keys=True).encode()).hexdigest()
    raw=np.array([s['features'] for s in samples]);mean=raw[train].mean(0);std=np.maximum(raw[train].std(0),1e-10)
    response=np.array([s['response'] for s in samples]);rmean=response[train].mean(0);rstd=np.maximum(response[train].std(0),1e-10)
    specs={};z=np.zeros((len(samples),2))
    for key in sorted(set(s['domain_id'] for s in samples)):
        indices=[i for i,s in enumerate(samples) if s['domain_id']==key]
        fitted=[i for i in indices if samples[i]['split']=='train']
        values=np.log1p(np.array([samples[i]['target'] for i in fitted]));mu=values.mean(0);sd=values.std(0)
        if np.any(sd<=0):raise ValueError('Degenerate native normalization')
        for i in indices:z[i]=(np.log1p(samples[i]['target'])-mu)/sd
        s=samples[indices[0]];specs[key]=dict(domain=s['domain'],parameters=s['parameters'],transforms=['log1p','log1p'],
            mean=mu.tolist(),std=sd.tolist(),fit_split='train',fit_manifest_sha256=train_hash)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    x=torch.tensor((raw-mean)/std,dtype=torch.float32,device=device)
    y=torch.tensor(z,dtype=torch.float32,device=device)
    r=torch.tensor((response-rmean)/rstd,dtype=torch.float32,device=device)
    domains=[s['domain_id'] for s in samples]
    train_t=torch.tensor(train,device=device);val_t=torch.tensor(val,device=device)
    pairs={}
    for i in train:pairs.setdefault((samples[i]['fixture'],samples[i]['material']),[]).append(int(i))
    if any(len(v)!=2 for v in pairs.values()):raise ValueError('Incomplete training pair')
    left=torch.tensor([v[0] for v in pairs.values()],device=device);right=torch.tensor([v[1] for v in pairs.values()],device=device)
    run_results=[]
    for seed in (17,29,43):
        for variant in ('native_only','response_aux','paired_alignment'):
            wait_if_paused(output.parent)
            torch.manual_seed(seed);model=Pilot(x.shape[1],specs).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            best=float('inf');bad=0;best_state=None;history=[];start=time.time()
            def evaluate(indices):
                latent,pred=model(x[indices],[domains[int(i)] for i in indices])
                native=torch.zeros((len(indices),2),device=device)
                for group in pred['native'].values():native[group['indices']]=group['mean_z']
                return latent,pred,native
            for epoch in range(1,701):
                wait_if_paused(output.parent);model.train();optimizer.zero_grad()
                latent,pred,native=evaluate(train_t)
                loss=(native-y[train_t]).square().mean()
                if variant!='native_only':loss=loss+.1*(pred['response']-r[train_t]).square().mean()
                if variant=='paired_alignment':
                    # Only identical calibrated fixture/material/load pairs in TRAIN.
                    loss=loss+.1*(model.encoder(x[left])-model.encoder(x[right])).square().mean()
                loss.backward();optimizer.step();model.eval()
                with torch.no_grad():
                    _,vp,vn=evaluate(val_t);score=float((vn-y[val_t]).abs().mean())
                history.append(dict(epoch=epoch,train_loss=float(loss.detach()),val_native_mae_z=score))
                if score<best-1e-4:best=score;best_epoch=epoch;bad=0;best_state=copy.deepcopy(model.state_dict())
                else:bad+=1
                if epoch%20==0:
                    atomic(output/'progress.json',dict(seed=seed,variant=variant,epoch=epoch,best_mae_z=best,bad_epochs=bad,updated_unix=time.time()))
                if epoch>=80 and bad>=80:break
            model.load_state_dict(best_state);model.eval()
            with torch.no_grad():
                _,pred,native=evaluate(val_t)
                native_np=native.cpu().numpy();response_np=pred['response'].cpu().numpy()*rstd+rmean
            per_engine={}
            for engine in ('warp','uipc'):
                ids=[j for j,i in enumerate(val) if samples[i]['engine']==engine]
                key=samples[int(val[ids[0]])]['domain_id'];spec=specs[key]
                decoded=np.expm1(native_np[ids]*np.array(spec['std'])+np.array(spec['mean']))
                truth=np.array([samples[int(val[j])]['target'] for j in ids])
                per_engine[engine]=dict(standardized_mae=float(np.abs(native_np[ids]-z[val[ids]]).mean()),
                    native_parameter_mape={name:float(np.mean(np.abs(decoded[:,k]/truth[:,k]-1))) for k,name in enumerate(spec['parameters'])})
            result=dict(seed=seed,variant=variant,best_epoch=best_epoch,epochs=epoch,val_native_mae_z=best,
                stop_reason='validation_plateau' if bad>=80 else 'epochs_complete',per_engine=per_engine,
                held_response_mae_m=float(np.abs(response_np-response[val]).mean()) if variant!='native_only' else None,
                wall_seconds=time.time()-start)
            identifier=f'{variant}_seed{seed}'
            atomic(output/(identifier+'.json'),{'result':result,'history':history})
            torch.save({'model':best_state,'specifications':specs,'feature_mean':mean,'feature_std':std,
                        'response_mean':rmean,'response_std':rstd,'train_manifest_sha256':train_hash},output/(identifier+'.pt'))
            run_results.append(result)
            atomic(ROOT/'docs/G2_ADAPTER_RESULTS.json',{'status':'running','runs':run_results,'updated_unix':time.time()})
    aggregate={v:dict(mean_val_native_mae_z=float(np.mean([r['val_native_mae_z'] for r in run_results if r['variant']==v])),
                      std_val_native_mae_z=float(np.std([r['val_native_mae_z'] for r in run_results if r['variant']==v])))
               for v in ('native_only','response_aux','paired_alignment')}
    atomic(ROOT/'docs/G2_ADAPTER_RESULTS.json',dict(status='complete',runs=run_results,aggregate=aggregate,
        samples={'train':len(train),'val':len(val),'test':0},train_manifest_sha256=train_hash,
        scope='Privileged full nodal X/Y displacement fields plus known geometry/load; predict native coefficients and held Z response. Same encoder/head capacity/initialization/budget, 3 seeds. Validation tunes/selects checkpoints; NOT final testing or video/no-bridge/online-fitting comparison.',
        leakage_controls='Fit all feature/response/native statistics on train only; entire geometry/material/engine/action groups stay in one split; engine ID selects native output head only.',
        updated_unix=time.time()))


if __name__=='__main__':main()
