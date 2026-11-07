import hashlib
import json
from pathlib import Path

import pytest

from fieldbridge.config import load_config
from fieldbridge.data import read_manifest
from fieldbridge.paths import path_map, relocate_path


def test_migration_preserves_manifest_digest_and_split_groups(tmp_path, monkeypatch):
    old, new = tmp_path/'old', tmp_path/'new'
    mapping = tmp_path/'mapping.json'
    mapping.write_text(json.dumps({str(old): str(new)}))
    monkeypatch.setenv('FIELDBRIDGE_PATH_MAP', str(mapping))
    manifest = tmp_path/'manifest.jsonl'
    row = dict(sample_id='a', object_id='obj', split='train', source='fixture',
               pack=str(old/'sample.npz'), parameters={})
    manifest.write_text(json.dumps(row)+'\n')
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert read_manifest(manifest)[0]['pack'] == str(new/'sample.npz')
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == checksum
    row.update(sample_id='b', split='val')
    with manifest.open('a') as f:
        f.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='leakage'):
        read_manifest(manifest)


def test_relocation_uses_directory_boundary_and_most_specific_rule(tmp_path):
    origin = tmp_path/'old'
    mappings = {str(origin): str(tmp_path/'new'), str(origin/'cache'): str(tmp_path/'fast')}
    assert relocate_path(str(origin/'cache/a.pt'), mappings) == str(tmp_path/'fast/a.pt')
    assert relocate_path(str(tmp_path/'older/a.pt'), mappings) == str(tmp_path/'older/a.pt')
    assert relocate_path('data/manifest.jsonl', mappings) == 'data/manifest.jsonl'
    assert relocate_path(str(origin/'../outside.npz'), mappings) == str(origin/'../outside.npz')


def test_invalid_map_fails_before_training_and_config_uses_mapping(tmp_path, monkeypatch):
    mapping = tmp_path/'mapping.json'
    mapping.write_text(json.dumps({'relative': str(tmp_path)}))
    monkeypatch.setenv('FIELDBRIDGE_PATH_MAP', str(mapping))
    with pytest.raises(ValueError, match='absolute'):
        path_map()
    old, new = tmp_path/'old', tmp_path/'new'
    mapping.write_text(json.dumps({str(old): str(new)}))
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'data': {'prepared_dir': str(old/'cache')},
                                  'train': {'resume': str(old/'last.pt')}}))
    loaded = load_config(config)
    assert loaded.data.prepared_dir == str(new/'cache')
    assert loaded.train.resume == str(new/'last.pt')
