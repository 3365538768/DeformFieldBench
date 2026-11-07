"""Freeze a 512/128 object-isolated diagnostic protocol and the optimization queue."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
from fieldbridge.config import load_config
from fieldbridge.data import read_manifest, fit_stats
from fieldbridge.runtime import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def select(rows, count, seed):
    rng = random.Random(seed); groups = defaultdict(list)
    for r in rows:
        groups[(r['object_id'], r.get('law_id'), r.get('action'))].append(r)
    keys = sorted(groups); rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < count:
        changed = False
        for key in keys:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop()); changed = True
        if not changed:
            raise ValueError('Insufficient samples')
    return chosen


def main():
    plan_path = ROOT / 'docs/OPTIMIZATION_PLAN.json'
    if plan_path.exists():
        raise FileExistsError('Immutable optimization plan already exists')
    rows = read_manifest('data/manifests/deformfield_v1.jsonl')
    diagnostic = select([r for r in rows if r['split'] == 'train'], 512, 17) + select([r for r in rows if r['split'] == 'val'], 128, 19)
    manifest = Path('data/manifests/g1_diagnostic_512_128.jsonl')
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in diagnostic))
    stats = manifest.with_suffix('.stats.json'); fit_stats(manifest, stats)
    split = {s: sum(r['split'] == s for r in diagnostic) for s in ['train', 'val', 'test']}
    atomic_json(manifest.with_suffix('.split.json'), {'split_counts': split,
        'objects': {s: len({r['object_id'] for r in diagnostic if r['split'] == s}) for s in ['train', 'val']},
        'manifest_sha256': sha256(manifest), 'test_excluded': True})
    base = json.loads(Path('configs/pilot_C_aux_no_feedback.json').read_text())
    base['data'].update(manifest=str(manifest), stats=str(stats), prepared_dir='data/prepared/g1_dinos16', view_dropout=0.)
    base['train'].update(epochs=24, field_epochs=2, batch_size=4, accumulation=2, workers=2,
        val_batch_size=4, val_workers=2, val_bf16=True, early_stop_patience=4,
        early_stop_min_joint_epochs=6, early_stop_relative_delta=.002, field_plateau_patience=3,
        purpose='diagnostic', log_interval=20, compact_checkpoint=True, resume='', initialize='')
    jobs = []
    for name, feedback, weight in [('C_aux_no_feedback', 'none', 1.), ('D_field_bridge', 'fields', 1.),
        ('G_feature_control', 'features', 1.), ('O_oracle_fields', 'oracle', 1.),
        ('B_rgb_only', 'none', 0.), ('S_shuffled_oracle', 'oracle_shuffled', 1.)]:
        cfg = copy.deepcopy(base); cfg['model']['feedback'] = feedback; cfg['train']['field_weight'] = weight
        run_id = 'g1_' + name + '_seed17'
        cfg['train']['output'] = 'runs/' + run_id
        config = 'configs/' + run_id + '.json'; atomic_json(config, cfg); load_config(config)
        jobs.append({'id': run_id, 'name': name, 'stage': 'G1', 'kind': 'train', 'config': config,
                     'devices': '', 'split_counts': split, 'log': f'runs/phase_logs/{run_id}.log',
                     'output': cfg['train']['output'], 'state': 'queued', 'desired': 'running'})
    for letter, name in [('C', 'C_aux_no_feedback'), ('D', 'D_field_bridge')]:
        original = f'configs/full_giant16_{name}_seed17.json'
        cfg = json.loads(Path(original).read_text())
        cfg['train'].update(output=f'runs/full_giant16_v2_{name}_seed17',
            resume=f'runs/full_giant16_{name}_seed17/last.pt', val_batch_size=2, val_workers=2,
            val_bf16=True, log_interval=20, early_stop_patience=5, early_stop_min_joint_epochs=6,
            early_stop_relative_delta=.002, field_plateau_patience=3, compact_checkpoint=True)
        config = f'configs/full_giant16_v2_{name}_seed17.json'; atomic_json(config, cfg); load_config(config)
        jobs.append({'id': Path(cfg['train']['output']).name, 'name': name + '_v2', 'stage': 'FULL', 'kind': 'train',
            'config': config, 'devices': '0,1,2,3' if letter == 'C' else '4,5,6,7',
            'split_counts': {'train': 4021, 'val': 399, 'test': 494}, 'log': f'runs/phase_logs/full_v2_{letter}.log',
            'output': cfg['train']['output'], 'state': 'held_for_g1', 'desired': 'running'})
    ck = 'runs/full_giant16_C_aux_no_feedback_seed17/last.pt'
    tasks = []
    for name, devices, extra in [('gold_fp32', '0', ['--batch', '1', '--workers', '0']),
                                  ('distributed_fp32', '0,1,2,3', ['--batch', '2', '--workers', '2']),
                                  ('distributed_bf16', '0,1,2,3', ['--batch', '2', '--workers', '2', '--bf16'])]:
        tasks.append({'id': name, 'stage': 'G0', 'kind': 'benchmark', 'devices': devices, 'state': 'queued',
            'args': ['scripts/benchmark_validation.py', '--config', 'configs/full_giant16_C_aux_no_feedback_seed17.json',
                     '--checkpoint', ck, '--output', f'runs/g0_benchmark/{name}', '--limit', '64', *extra],
            'log': f'runs/phase_logs/g0_{name}.log'})
    cache_config = 'configs/g1_C_aux_no_feedback_seed17.json'
    tasks.append({'id': 'g1_cache', 'stage': 'G1', 'kind': 'cache', 'devices': '0', 'state': 'queued',
                  'args': ['scripts/prepare_feature_cache.py', '--config', cache_config],
                  'log': 'runs/phase_logs/g1_cache.log'})
    atomic_json(plan_path, {'version': 2, 'jobs': jobs, 'tasks': tasks,
        'feedback_nodes': ['G0 数值/覆盖/耗时验收', 'G1 真场上界与等容量诊断', 'G2 引擎标定与新动作响应',
                           'G3 新数据和局部时序桥', 'G4 三种子强基线'],
        'automatic_full_gate': 'oracle and predicted bridge each improve C by at least 2%; provisional routing, not significance',
        'stop_policy': {'relative_delta': .002, 'joint_patience_full': 5, 'joint_patience_diagnostic': 4,
                        'min_joint_epochs': 6, 'field_plateau_ends_warmup': True}})
    atomic_json('docs/OPTIMIZATION_CONTROL.json', {'desired': 'running'})
    print(json.dumps({'split_counts': split, 'jobs': len(jobs), 'tasks': len(tasks)}))


if __name__ == '__main__':
    main()
