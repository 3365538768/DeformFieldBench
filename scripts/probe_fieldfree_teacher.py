"""Known-force projected-track inverse-FEM teacher -> amortized student check.

This fixture is not an RGB-video student and does not admit original MPM labels.
Only E is inferred; geometry, nu, rho, loading and camera scale are known.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
import torch
from torch import nn
from fieldbridge.linear_fixture import assemble,equilibrium
from engine_fixture import atomic,block_mesh,tip_weights,wait_if_paused

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);torch.set_num_threads(1);torch.manual_seed(17);rng=np.random.default_rng(17)
    out=ROOT/'runs/nf_teacher_fixture';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=120,test_untouched=True,
        detail='受控线性FEM：已知力/尺度/几何，投影轨迹逆拟合E后蒸馏学生；不是原GS视频训练或真实材参结果。',
        scope='Projected-track and known-force feature student, not an RGB model. Fixed nu=.3/rho=1000; no contact/plasticity; no cross-engine equivalence claim.')
    def update(**values):
        report.update(values,updated_unix=time.time());atomic(ROOT/'docs/NF_TEACHER_FIXTURE.json',report)
    try:
        quality=json.loads((ROOT/'docs/I1_IDENTIFIABILITY.json').read_text())
        if not quality['admissions']['reference_fixture_quality']:raise ValueError('Controlled fixture not admitted')
        rest,tets=block_mesh([4,2,2],[.12,.04,.04]);fixed=np.flatnonzero(np.repeat(rest[:,0]==0,3));clamp=dict.fromkeys(fixed,0.)
        tip=tip_weights(rest,tets,.12);k,_=assemble(rest,tets,10000.,.3,1000.)
        force=np.zeros_like(rest);force[:,2]=tip*.1
        unit,_,residual=equilibrium(k,force.flatten(),clamp)
        # Orthographic camera sees X and Z, with explicit metric scale.
        pixels_per_metre=1600.;basis=unit.reshape(-1,3)[:,[0,2]]*pixels_per_metre
        profile=np.sin(np.linspace(0,np.pi/2,16))**2
        rows=[];forward_calls=0
        for i in range(120):
            wait_if_paused(out);young=float(np.exp(rng.uniform(np.log(3000),np.log(30000))));load=float(rng.uniform(.02,.1))
            exact=profile[:,None,None]*basis[None]*(10000/young)*(load/.1)
            observed=exact+rng.normal(0,.002,exact.shape)
            def loss(loge):
                nonlocal forward_calls
                forward_calls+=1
                predicted=profile[:,None,None]*basis[None]*(10000/np.exp(loge))*(load/.1)
                return float(np.mean((predicted-observed)**2))
            fit=minimize_scalar(loss,bounds=(np.log(1000),np.log(100000)),method='bounded',options={'xatol':1e-9})
            if not fit.success:raise ValueError('Inverse teacher failed')
            amplitude=float(np.sum(observed*(profile[:,None,None]*basis[None]))/np.sum((profile[:,None,None]*basis[None])**2))
            if amplitude<=0:raise ValueError('Non-positive observed response')
            # Compact observation statistics and known loading, not material GT.
            x=[np.log(amplitude),np.log(load/.1)]
            rows.append(dict(x=x,teacher_log_E=float(fit.x),ground_truth_E=young,load_n=load,
                             split='train' if i<96 else 'val',teacher_response_mse=float(fit.fun)))
            update(completed=i+1)
        x=torch.tensor([r['x'] for r in rows],dtype=torch.float32);y=torch.tensor([r['teacher_log_E'] for r in rows],dtype=torch.float32)
        center=x[:96].mean(0);scale=x[:96].std(0).clamp_min(.1);z=(x-center)/scale
        student=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,1));optimizer=torch.optim.AdamW(student.parameters(),lr=.01,weight_decay=0.)
        history=[]
        for epoch in range(400):
            wait_if_paused(out);optimizer.zero_grad();loss=(student(z[:96]).squeeze(-1)-y[:96]).square().mean();loss.backward();optimizer.step()
            if epoch%50==0:history.append(dict(epoch=epoch+1,train_log_E_mse=float(loss.detach())))
        with torch.no_grad():prediction=student(z).squeeze(-1).numpy()
        truth=np.array([r['ground_truth_E'] for r in rows]);teacher=np.exp(np.array([r['teacher_log_E'] for r in rows]));student_E=np.exp(prediction)
        val=slice(96,None);teacher_error=float(np.mean(np.abs(teacher[val]/truth[val]-1)));student_error=float(np.mean(np.abs(student_E[val]/truth[val]-1)))
        # Unseen load amplitudes .5/1.5 times each observed action; linear prediction
        # response error is inverse-E error. This is same-solver validation only.
        response_error=float(np.mean(np.abs(truth[val]/student_E[val]-1)))
        torch.save(dict(model=student.state_dict(),center=center,scale=scale),out/'student.pt');atomic(out/'teacher_records.json',rows)
        update(status='complete',free_equation_residual=residual,train_samples=96,val_samples=24,
               teacher_val_mean_relative_E_error=teacher_error,student_val_mean_relative_E_error=student_error,
               unseen_load_student_mean_relative_response_error=response_error,teacher_forward_calls=forward_calls,student_updates=400,
               training_history=history,pixels_per_metre=pixels_per_metre,observation_noise_std_px=.002,
               pipeline_gate_passed=teacher_error<.01 and student_error<.1,
               original_video_teacher_admitted=False,rgb_video_distillation_completed=False,
               limitations=['Highly simplified known-force, known-geometry single-parameter fixture.',
                            'Labels used by student are optimizer estimates; true E used only to generate and evaluate fixture.',
                            'Projected tracks are privileged inputs, not automatically available from monocular video.'])
        atomic(out/'status.json',dict(status='complete',reason='controlled_teacher_student_pipeline_checked'))
    except Exception as error:
        update(status='failed',error=str(error));atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
