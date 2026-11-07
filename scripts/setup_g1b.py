"""Register the bounded, validation-only G1b loss/gradient diagnostic matrix."""
import argparse
import copy
import json
from pathlib import Path
from optimization_manager import ROOT, atomic, merge_jobs, read


MATRIX = [
    ('B_rgb_repeat', 'none', 0., False, 'bce'),
    ('C_weight01', 'none', .1, False, 'bce'),
    ('D_weight01', 'fields', .1, False, 'bce'),
    ('G_weight01', 'features', .1, False, 'bce'),
    ('C_weight001', 'none', .01, False, 'bce'),
    ('D_weight001', 'fields', .01, False, 'bce'),
    ('D_weight01_detached', 'fields', .1, True, 'bce'),
    ('C_weight01_balanced', 'none', .1, False, 'balanced_bce'),
    ('D_weight01_balanced', 'fields', .1, False, 'balanced_bce'),
]


def setup(register=False):
    base = read(ROOT / 'configs/g1_D_field_bridge_seed17.json')
    jobs = []
    for name, feedback, weight, detach, contact in MATRIX:
        identifier = f'g1b_{name}_seed17'
        cfg = copy.deepcopy(base)
        cfg['data']['prepared_dir'] = 'data/prepared/g1_dinos16_local'
        cfg['model'].update(feedback=feedback, detach_feedback=detach)
        cfg['train'].update(output=f'runs/{identifier}', field_weight=weight, contact_loss=contact, resume='', initialize='')
        # Preserve the cached encoder, manifest, optimizer, minibatches and seed.
        config = f'configs/{identifier}.json'
        existing = read(ROOT / config)
        if existing is not None and existing != cfg:
            raise ValueError(f'Refusing to overwrite a different configuration: {config}')
        atomic(ROOT / config, cfg)
        jobs.append(dict(id=identifier, name=name, stage='G1b', kind='train', config=config,
                         devices='', split_counts=dict(train=512, val=128, test=0),
                         log=f'runs/phase_logs/{identifier}.log', output=cfg['train']['output'],
                         state='queued', desired='running'))
    if register:
        original = read(ROOT / 'docs/G1_DIAGNOSTIC_RESULTS.json', {})
        if original.get('status') != 'complete':
            raise ValueError('G1 must be complete before registering G1b')
        plan = read(ROOT / 'docs/OPTIMIZATION_PLAN.json')
        plan['jobs'] = merge_jobs(plan['jobs'], jobs)
        plan['automatic_full_gate'] = 'No G1b automatic promotion; review matched B/C/G controls, then repeat independent seeds.'
        atomic(ROOT / 'docs/OPTIMIZATION_PLAN.json', plan)
    print(json.dumps({'registered': register, 'jobs': [j['id'] for j in jobs]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    setup(parser.parse_args().register)
