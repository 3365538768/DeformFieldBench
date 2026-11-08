"""Controlled linear-FEM loading/scale audit, cross-checked against Warp FEM.

No DeformFieldBench labels are changed. Findings apply to this fixture and its
assumptions; native MPM, finite strain and plastic/contact cases need new audits.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
from fieldbridge.linear_fixture import assemble,equilibrium,frequencies
from fieldbridge.runtime import sha256
from engine_fixture import atomic,block_mesh,tip_weights,wait_if_paused,run_controlled_solver

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);out=ROOT/'runs/i1_identifiability_solve64';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=8,cases=[],test_untouched=True,
        detail='已知小应变夹具：定力/定位移、E/密度尺度多解、ν敏感性；独立线性FEM与Warp交叉校验。',
        scope='Controlled isotropic linear elastic block only. Not a replay of the original MPM dataset, not proof of plastic/finite-strain equivalence, and not a new material ground-truth source.')
    def update(**kwargs):
        report.update(kwargs,updated_unix=time.time());atomic(ROOT/'docs/I1_IDENTIFIABILITY.json',report)
    try:
        case=dict(cells=[4,2,2],dimensions_m=[.12,.04,.04],young_pa=10000.,poisson=.3,density_kg_m3=1000.,force_n=.002,axis=2,device='cpu',solver_precision='float64')
        update(retry_of=['runs/i1_identifiability','runs/i1_identifiability_f64','runs/i1_identifiability_verified'],
            repair='FP64 integration accumulation, matrix storage and solve; FP32 Warp geometry/basis/form values; same admission gates. Prior FP32 and mixed-basis failures retained.')
        rest,tets=block_mesh(case['cells'],case['dimensions_m']);weights=tip_weights(rest,tets,.12)
        fixed=np.flatnonzero(np.repeat(rest[:,0]==0,3));tip=np.flatnonzero(np.isclose(rest[:,0],.12))
        clamp=dict.fromkeys(fixed,0.);displacement=dict(clamp)
        for node in tip:
            for axis in range(3):displacement[3*node+axis]=.0001 if axis==2 else 0.
        force=np.zeros_like(rest);force[:,2]=weights*case['force_n'];force=force.flatten()
        # Fixed protocol before examining outcomes, with no quality-based filtering.
        specifications=[('reference',1.,1.,.3,1.),('E_half',.5,1.,.3,1.),('E_double',2.,1.,.3,1.),
            ('rho_double',1.,2.,.3,1.),('E_rho_force_double',2.,2.,.3,2.),('nu_lower',1.,1.,.25,1.),('nu_higher',1.,1.,.35,1.)]
        arrays={};maximum_residual=0.
        for name,e_scale,rho_scale,nu,f_scale in specifications:
            wait_if_paused(out)
            k,m=assemble(rest,tets,case['young_pa']*e_scale,nu,case['density_kg_m3']*rho_scale)
            uf,rf,ef=equilibrium(k,force*f_scale,clamp)
            ud,rd,ed=equilibrium(k,np.zeros_like(force),displacement)
            frequency=frequencies(k,m,fixed);maximum_residual=max(maximum_residual,ef,ed)
            arrays[name]=dict(force_displacement=uf,dirichlet_displacement=ud,frequency=frequency)
            report['cases'].append(dict(name=name,young_pa=case['young_pa']*e_scale,density_kg_m3=case['density_kg_m3']*rho_scale,
                poisson=nu,force_n=case['force_n']*f_scale,
                force_control_tip_displacement_m=(weights@uf.reshape(-1,3)).tolist(),
                displacement_control_tip_reaction_n=rd.reshape(-1,3)[tip].sum(0).tolist(),
                first_three_natural_frequencies_hz=frequency.tolist(),free_relative_residual=max(ef,ed)))
            np.savez_compressed(out/(name+'.npz'),rest_positions_m=rest,tetrahedra=tets,**arrays[name])
            update(completed=len(report['cases']))
        wait_if_paused(out);atomic(out/'case.json',case)
        command=[str(ROOT/'.engine-envs/warp/bin/python'),'scripts/solve_engine_fixture.py','--case',str(out/'case.json'),
            '--engine','warp','--output',str(out/'warp_reference')]
        with (out/'warp.log').open('w') as log:
            run_controlled_solver(command,log,out,180,ROOT)
        warp=json.loads((out/'warp_reference/result.json').read_text())
        with np.load(out/'warp_reference/field.npz') as archive:
            if not np.allclose(archive['rest_positions_m'],rest):raise ValueError('Cross-solver node ordering mismatch')
            warp_u=archive['displacement_m'].flatten()
        reference=arrays['reference'];relative=float(np.linalg.norm(warp_u-reference['force_displacement'])/np.linalg.norm(reference['force_displacement']))
        cross_pass=relative<.005 and warp['actual_relative_residual']<1e-4 and maximum_residual<1e-8
        metrics={}
        for name,data in arrays.items():
            metrics[name]={key:float(np.linalg.norm(data[key]-reference[key])/np.linalg.norm(reference[key])) for key in data}
        expected=dict(force_E_half_relative_error=float(np.linalg.norm(arrays['E_half']['force_displacement']/2-reference['force_displacement'])/np.linalg.norm(reference['force_displacement'])),
            force_E_double_relative_error=float(np.linalg.norm(arrays['E_double']['force_displacement']*2-reference['force_displacement'])/np.linalg.norm(reference['force_displacement'])))
        admitted=cross_pass and max(expected.values())<1e-8 and metrics['E_double']['dirichlet_displacement']<1e-8 and metrics['E_rho_force_double']['frequency']<1e-8
        update(status='complete' if admitted else 'failed',completed=8,controlled_fixture=case,
            cross_solver=dict(reference='float64 P1 tetra / scipy',check='Warp FEM P1 on CPU: FP32 geometry/basis/forms, FP64 accumulation/storage/solve',
                relative_displacement_difference=relative,gate=.005,passed=cross_pass,warp_result_sha256=sha256(out/'warp_reference/result.json')),
            normalized_response_changes=metrics,scaling_checks=expected,maximum_free_residual=maximum_residual,
            admissions=dict(reference_fixture_quality=admitted,original_mpm_replay=False,contact_force_ground_truth=False,mass_dataset_generation=False),
            findings=['Under displacement control, homogeneous E scaling leaves this static displacement unchanged while reactions scale.',
                'Under known force, static displacement scales inversely with E; density is absent from static equilibrium.',
                'Joint E/rho/unknown-force scaling leaves static force response and natural frequencies invariant in this linear fixture.',
                'Nu changes response here; sensitivity does not establish unique recoverability from a monocular video.'],
            next_step='Use calibrated loading/scale metadata or report response equivalence and uncertainty. Validate original MPM sensitivity separately before absolute-parameter claims.')
        atomic(out/'status.json',dict(status=report['status'],reason='controlled_linear_identifiability_audit'))
        if not admitted:raise ValueError('Controlled fixture numerical checks failed')
    except Exception as error:
        update(status='failed',error=str(error));atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
