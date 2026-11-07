"""Replay native predictions under held-out Z loading in each actual engine.

Observed X/Y full displacement, mesh/scale/load are privileged for ALL methods.
Classical reference is closed-form linear inverse FEM, not an RGB benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from engine_fixture import atomic, tip_weights, wait_if_paused
from run_cross_engine_pilot import ROOT, solve
from train_engine_adapter_pilot import Pilot, arrays


def inverse_fem(actions):
    columns=[];loads=[]
    for row in actions:
        with np.load(ROOT/row['path']/'field.npz',allow_pickle=False) as data:
            rest=data['rest_positions_m'];u=data['displacement_m'];tets=data['tetrahedra']
        basis=np.stack([rest[tets[:,i]]-rest[tets[:,0]] for i in (1,2,3)],-1)
        gradients=np.concatenate([-np.linalg.inv(basis).sum(axis=1,keepdims=True),np.linalg.inv(basis)],axis=1)
        volume=np.linalg.det(basis)/6
        gu=np.einsum('tni,tnj->tij',u[tets],gradients)
        eps=.5*(gu+gu.transpose(0,2,1))
        sigma_mu=2*eps;sigma_lam=np.trace(gu,axis1=1,axis2=2)[:,None,None]*np.eye(3)
        forces=[]
        for sigma in (sigma_mu,sigma_lam):
            element=np.einsum('tij,tnj,t->tni',sigma,gradients,volume)
            nodal=np.zeros_like(rest);np.add.at(nodal,tets.reshape(-1),element.reshape(-1,3));forces.append(nodal)
        applied=np.zeros_like(rest);applied[:,row['case']['axis']]=tip_weights(rest,tets,row['case']['dimensions_m'][0])*row['case']['force_n']
        free=rest[:,0]!=0
        columns.append(np.stack([f[free].reshape(-1) for f in forces],-1));loads.append(applied[free].reshape(-1))
    a=np.concatenate(columns);b=np.concatenate(loads);value,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
    if rank!=2:raise ValueError('Inverse FEM coefficients not identifiable')
    return value,float(np.linalg.norm(a@value-b)/np.linalg.norm(b))


def native_to_enu(mu,second,engine):
    lam=second if engine=='warp' else second-mu
    if not np.isfinite([mu,lam]).all() or mu<=0 or lam<0:
        raise ValueError('Predicted coefficients outside calibrated positive elastic domain')
    nu=lam/(2*(lam+mu));young=mu*(3*lam+2*mu)/(lam+mu)
    if not 0<=nu<.49:raise ValueError('Predicted Poisson ratio outside supported range')
    return float(young),float(nu)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/g2_engine_replay');args=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(1);output=ROOT/args.output
    manifest=json.loads((ROOT/'data/manifests/g2_engine_response.json').read_text());samples=arrays(manifest)
    val=[s for s in samples if s['split']=='val'];groups={}
    for row in manifest['rows']:groups.setdefault((row['fixture'],row['material_pair'],row['engine']),{})[row['axis']]=row
    tasks=[]
    for variant in ('native_only','response_aux','paired_alignment','train_mean','inverse_fem'):
        checkpoint=torch.load(ROOT/f'runs/g2_cross_engine_pilot/adapter/{variant if variant not in {"train_mean","inverse_fem"} else "native_only"}_seed17.pt',map_location='cpu',weights_only=False)
        predictions=np.zeros((len(val),2));inverse_residuals={}
        if variant not in {'train_mean','inverse_fem'}:
            model=Pilot(len(checkpoint['feature_mean']),checkpoint['specifications']);model.load_state_dict(checkpoint['model']);model.eval()
            x=torch.tensor((np.array([s['features'] for s in val])-checkpoint['feature_mean'])/checkpoint['feature_std'],dtype=torch.float32)
            with torch.no_grad():_,result=model(x,[s['domain_id'] for s in val])
            for item in result['native'].values():predictions[item['indices'].numpy()]=item['mean_z'].numpy()
        for i,s in enumerate(val):
            wait_if_paused(output);actions=groups[(s['fixture'],s['material'],s['engine'])]
            if variant=='inverse_fem':
                coeff,residual=inverse_fem([actions[0],actions[1]]);mu,lam=coeff
                native=[float(mu),float(lam if s['engine']=='warp' else lam+mu)]
                inverse_residuals[i]=residual
            else:
                spec=checkpoint['specifications'][s['domain_id']]
                native=np.expm1(predictions[i]*np.array(spec['std'])+np.array(spec['mean'])).tolist()
            young,nu=native_to_enu(*native,s['engine'])
            # Reuse only nuisance fixture/load/solver metadata. Replace both GT elastic parameters.
            case={**actions[2]['case'],'young_pa':young,'poisson':nu}
            tasks.append(dict(variant=variant,seed=17,engine=s['engine'],fixture=s['fixture'],
                material_pair=s['material'],prediction_native_pa=native,truth_native_pa=s['target'],case=case,
                truth_path=actions[2]['path'],inverse_observation_residual=inverse_residuals.get(i)))
    total=len(tasks);results=[];started=time.time()
    def progress(status='running'):
        value=dict(status=status,completed=len(results),total=total,updated_unix=time.time(),
            detail=f'预测原生参数回放未见 Z 载荷：{len(results)}/{total}；含训练均值及同观测经典逆 FEM。',
            scope='seed17, two held geometries, observed full X/Y displacement fields and known geometry/load for all. Not video inference or an existing-paper online fitting benchmark.',
            results=results)
        if status=='complete':
            value['aggregate']={v:{engine:dict(mean_relative_field_l2=float(np.mean([r['relative_field_l2'] for r in results if r['variant']==v and r['engine']==engine])),
                median_relative_field_l2=float(np.median([r['relative_field_l2'] for r in results if r['variant']==v and r['engine']==engine])))
                for engine in ('warp','uipc')} for v in ('native_only','response_aux','paired_alignment','train_mean','inverse_fem')}
            value['wall_seconds']=time.time()-started
        atomic(ROOT/'docs/G2_ENGINE_REPLAY.json',value)
        atomic(output/'runtime.json',dict(phase=status,completed=len(results),total=total,updated_unix=time.time()))
    progress()
    def worker(task):
        wait_if_paused(output);key,receipt=solve(task['engine'],task['case'],ROOT/'data/calibration/g2_replay')
        if not receipt['completed'] or receipt['min_jacobian']<=0:raise ValueError('Failed replay solve')
        predicted=np.load(ROOT/'data/calibration/g2_replay'/key/'field.npz',allow_pickle=False)['displacement_m']
        truth=np.load(ROOT/task['truth_path']/'field.npz',allow_pickle=False)['displacement_m']
        return {k:v for k,v in task.items() if k!='case'}|dict(predicted_case_id=key,
            relative_field_l2=float(np.linalg.norm(predicted-truth)/np.linalg.norm(truth)),
            field_mae_m=float(np.abs(predicted-truth).mean()),solver_seconds=receipt['wall_seconds'])
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(worker,t) for t in tasks]):
                results.append(future.result());progress()
        progress('complete');atomic(output/'status.json',dict(status='complete',reason='held_action_replay_complete'))
    except Exception as exc:
        atomic(output/'status.json',dict(status='failed',reason=str(exc)));raise


if __name__=='__main__':main()
