import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from fieldbridge.config import Config, ModelConfig
from fieldbridge.data import MaterialDataset, collate, fit_stats, model_inputs
from fieldbridge.model import MaterialFieldModel
from fieldbridge.runtime import Plateau, atomic_json, checkpoint_weights, load_weights
import fieldbridge.train as trainer

spec = importlib.util.spec_from_file_location('fixtures_v2', Path(__file__).parents[1] / 'scripts/make_fixture.py')
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


def configuration(tmp_path):
    torch.set_num_threads(1)
    manifest = fixtures.create(tmp_path / 'data')
    cfg = Config(model=ModelConfig(width=16, heads=4, layers=1, field_size=8))
    cfg.data.manifest = str(manifest); cfg.data.stats = str(tmp_path / 'stats.json')
    cfg.data.frames = 4; cfg.data.image_size = 32
    cfg.train.device = 'cpu'; cfg.train.workers = cfg.train.val_workers = 0
    cfg.train.epochs = 3; cfg.train.field_epochs = 1; cfg.train.early_stop_patience = 0
    cfg.train.purpose = 'smoke'; cfg.train.log_interval = 20
    fit_stats(manifest, cfg.data.stats)
    return cfg


def test_plateau_requires_meaningful_improvement_and_minimum_observations():
    p = Plateau(2, .002)
    assert not p.observe(1.)
    assert not p.observe(.9999, 3)
    assert p.observe(.9998, 3)
    assert not p.observe(.9)
    assert p.state['bad_epochs'] == 0
    restored = Plateau(2, .002, copy.deepcopy(p.state))
    assert restored.state == p.state


def test_pause_resume_is_weight_identical_at_epoch_boundary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = configuration(tmp_path)
    config = tmp_path / 'config.json'
    cfg.train.output = str(tmp_path / 'baseline'); atomic_json(config, cfg.dump()); trainer.run(config)
    expected = torch.load(Path(cfg.train.output) / 'last.pt', weights_only=False)
    cfg.train.output = str(tmp_path / 'paused'); atomic_json(config, cfg.dump())
    evaluate = trainer.evaluate
    def pause_after_validation(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        atomic_json(Path(cfg.train.output) / 'control.json', {'desired': 'paused'})
        return result
    monkeypatch.setattr(trainer, 'evaluate', pause_after_validation); trainer.run(config)
    checkpoint = Path(cfg.train.output) / 'last.pt'
    paused = torch.load(checkpoint, weights_only=False)
    assert paused['epoch'] == 0 and not paused['validation_pending']
    assert not (Path(cfg.train.output) / 'best_material.pt').exists()
    assert json.loads((Path(cfg.train.output) / 'runtime.json').read_text())['phase'] == 'paused'
    atomic_json(Path(cfg.train.output) / 'control.json', {'desired': 'running'})
    cfg.train.resume = str(checkpoint); atomic_json(config, cfg.dump())
    monkeypatch.setattr(trainer, 'evaluate', evaluate); trainer.run(config)
    actual = torch.load(checkpoint, weights_only=False)
    assert actual['step'] == expected['step']
    for name in expected['model']:
        assert torch.equal(actual['model'][name], expected['model'][name]), name


def test_flat_joint_validation_stops_and_preserves_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); cfg = configuration(tmp_path)
    cfg.train.output = str(tmp_path / 'flat'); cfg.train.epochs = 12; cfg.train.field_epochs = 0
    cfg.train.lr = 0.; cfg.train.early_stop_patience = 2; cfg.train.early_stop_min_joint_epochs = 3
    config = tmp_path / 'flat.json'; atomic_json(config, cfg.dump()); trainer.run(config)
    status = json.loads((Path(cfg.train.output) / 'status.json').read_text())
    checkpoint = torch.load(Path(cfg.train.output) / 'last.pt', weights_only=False)
    assert status['status'] == 'early_stopped' and checkpoint['epoch'] == 2
    assert (Path(cfg.train.output) / 'best_material.pt').exists()


def test_cached_forward_and_oracle_access_control(tmp_path):
    cfg = configuration(tmp_path)
    ds = MaterialDataset(cfg.data, 'train', 8, 'legacy_visual'); b = collate([ds[0]])
    model = MaterialFieldModel(cfg.model).eval()
    with torch.no_grad():
        expected = model(**model_inputs(b))['mean']
        features = model.frame_encoder(b['rgb'].flatten(0, 2)).reshape(*b['rgb'].shape[:3], 64, 4, 4)
        b['features'] = features; b['rgb'] = b['rgb'][..., :1, :1]
        actual = model(**model_inputs(b))['mean']
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match='GT fields rejected'):
        model(**model_inputs(b), diagnostic_fields=b['fields'])
    cfg.model.feedback = 'oracle'
    with pytest.raises(ValueError, match='privileged'):
        cfg.validate()
    cfg.train.purpose = 'diagnostic'; cfg.validate()


def test_uneven_distributed_validation_has_exact_coverage(tmp_path):
    cfg = configuration(tmp_path)
    rows = [json.loads(x) for x in Path(cfg.data.manifest).read_text().splitlines()]
    val = next(r for r in rows if r['split'] == 'val')
    rows = [r for r in rows if r['split'] != 'val'] + [dict(val, sample_id=f'val{i}', object_id=f'val_object{i}') for i in range(5)]
    Path(cfg.data.manifest).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    fit_stats(cfg.data.manifest, cfg.data.stats)
    config = tmp_path / 'distributed.json'; atomic_json(config, cfg.dump())
    script = Path(__file__).with_name('validation_worker.py')
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                             str(script), str(config), str(tmp_path / 'val_results')],
                            capture_output=True, text=True, timeout=90, env={**os.environ, 'OMP_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
