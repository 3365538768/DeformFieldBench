"""Queue extensions must preserve finished jobs and never auto-promote tuning."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

pytest.importorskip('fcntl', reason='Persistent A800 queue runs on Linux')

spec = importlib.util.spec_from_file_location('manager', Path(__file__).parents[1] / 'scripts/optimization_manager.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


def test_append_does_not_reset_finished_or_paused_jobs():
    old = [{'id': 'a', 'state': 'finished', 'pid': 23}, {'id': 'b', 'state': 'paused'}]
    planned = [{'id': 'a', 'state': 'queued'}, {'id': 'b', 'state': 'queued'}, {'id': 'c', 'state': 'queued'}]
    snapshot = copy.deepcopy(old)
    merged = manager.merge_jobs(old, planned)
    assert merged == old + [planned[-1]]
    merged[0]['state'] = 'running'
    assert old == snapshot


def test_new_backbone_cannot_reuse_old_cache_or_unfinished_baseline():
    job = {'depends_on': ['v3_cache', 'v3_rgb']}
    historical = [{'id':'v2_cache','state':'finished'}, {'id':'v2_rgb','state':'finished'}]
    assert not manager.dependencies_finished(job, historical)
    jobs = historical + [{'id':'v3_cache','state':'finished'}, {'id':'v3_rgb','state':'running'}]
    assert not manager.dependencies_finished(job, jobs)
    jobs[-1]['state']='failed'
    assert not manager.dependencies_finished(job, jobs)
    jobs[-1]['state']='finished'
    assert manager.dependencies_finished(job, jobs)


def test_strong_backbone_uses_its_own_rgb_not_historical_rgb(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, 'ROOT', tmp_path)
    jobs=[]
    for name, score in [('B_rgb_only',.65), ('D_local_fields',.69), ('G_local_features',.7)]:
        output=tmp_path/name;output.mkdir()
        (output/'timing.jsonl').write_text(json.dumps(dict(phase='joint_training',epoch=1,val_mae=score))+'\n')
        jobs.append(dict(name=name,kind='train',stage='G1f',state='finished',output=name))
    result=manager.strong_backbone_results(jobs)
    assert result['original_rgb_mae']==.65
    assert not result['passes_2_percent_routing_threshold']
    assert not result['resume_full_pair']


def test_full_gate_includes_strong_rgb_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, 'ROOT', tmp_path)
    names = ['C_aux_no_feedback', 'D_field_bridge', 'G_feature_control', 'O_oracle_fields', 'B_rgb_only', 'S_shuffled_oracle']
    scores = [.9, .85, .89, .7, .8, .9]
    jobs = []
    for name, score in zip(names, scores):
        output = tmp_path / name
        output.mkdir()
        (output / 'timing.jsonl').write_text(json.dumps(dict(phase='joint_training', epoch=1, val_mae=score)) + '\n')
        jobs.append(dict(stage='G1', name=name, state='finished', output=name))
    result = manager.diagnostic_results(jobs)
    assert not result['resume_full_pair']  # D beats C/G, but is worse than B.
    jobs[1]['stage'] = 'G1b'
    followup = manager.followup_results([jobs[1]], result)
    assert followup['status'] == 'complete'
    assert not followup['resume_full_pair']


@pytest.mark.parametrize('suffix,score,expected', [
    ('weight01', .76, 'repeat_independent_seeds'),
    ('weight01', .81, 'repair_fields_and_local_temporal_bridge'),
    ('weight001', .76, 'add_matched_feature_control'),
])
def test_followup_routes_require_rgb_and_matching_controls(tmp_path, monkeypatch, suffix, score, expected):
    monkeypatch.setattr(manager, 'ROOT', tmp_path)
    scores = {'B_rgb_repeat': .8, 'C_' + suffix: .79, 'D_' + suffix: score}
    if suffix == 'weight01':
        scores['G_weight01'] = .8
    jobs = []
    for name, value in scores.items():
        output = tmp_path / name; output.mkdir()
        (output / 'timing.jsonl').write_text(json.dumps(dict(phase='joint_training', epoch=1, val_mae=value)) + '\n')
        jobs.append(dict(stage='G1b', name=name, state='finished', output=name))
    original = {'results': {'B_rgb_only': {'best_material_mae': .8}}}
    result = manager.followup_results(jobs, original)
    assert result['route'] == expected
    assert not result['resume_full_pair']


def test_local_bridge_epoch_zero_fallback_is_not_an_improvement(tmp_path, monkeypatch):
    monkeypatch.setattr(manager,'ROOT',tmp_path)
    jobs=[]
    for name,trained in [('C_frozen_rgb',.8),('D_local_fields',.84),('G_local_features',.79)]:
        output=tmp_path/name;output.mkdir()
        (output/'initial_evaluation.json').write_text(json.dumps({'standardized_mae':.8}))
        (output/'timing.jsonl').write_text(json.dumps(dict(phase='joint_training',epoch=1,val_mae=trained))+'\n')
        jobs.append(dict(name=name,stage='G1c',state='finished',output=name))
    original={'results':{'B_rgb_only':{'best_material_mae':.8}}}
    result=manager.local_bridge_results(jobs,original)
    assert result['results']['D_local_fields']['best_epoch']==0
    assert result['results']['D_local_fields']['best_material_mae']==.8
    assert not result['passes_2_percent_routing_threshold']
    assert result['route']=='review_field_fit_and_data_targets'
    assert not result['resume_full_pair']
