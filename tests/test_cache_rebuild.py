import importlib.util
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip('fcntl', reason='Cache rebuild supervisor runs on Linux')

spec = importlib.util.spec_from_file_location('cache_rebuild', Path(__file__).parents[1] / 'scripts/rebuild_feature_cache.py')
rebuild = importlib.util.module_from_spec(spec); spec.loader.exec_module(rebuild)


def test_rebuild_rejects_invalid_artifacts(tmp_path):
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'files': {'sample': 'sample.pt'}}))
    with pytest.raises(ValueError, match='IDs differ'):
        rebuild.verify_cache(tmp_path, ['different'], lambda *_: None)
    sample = dict(sample_id='sample', features=torch.full((1, 2, 4), float('nan')),
                  valid=torch.ones(1, 2), fields=torch.zeros(1), times=torch.zeros(2),
                  target=torch.zeros(4), prior=torch.zeros(4))
    torch.save(sample, tmp_path / 'sample.pt')
    with pytest.raises(ValueError, match='Invalid cached'):
        rebuild.verify_cache(tmp_path, ['sample'], lambda *_: None)
    sample['features'].zero_(); torch.save(sample, tmp_path / 'sample.pt')
    assert len(rebuild.verify_cache(tmp_path, ['sample'], lambda *_: None)['sample.pt']) == 64


def test_activation_preserves_partial_directory_and_rejects_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prepared = tmp_path / 'data/prepared'; prepared.mkdir(parents=True)
    partial = prepared / 'old'; partial.mkdir(); (partial / 'partial.pt').write_bytes(b'preserve')
    fresh = prepared / 'fresh'; fresh.mkdir()
    receipt = rebuild.activate_alias(partial, fresh)
    assert partial.is_symlink() and partial.resolve() == fresh
    assert (Path(receipt['retained_copy']) / 'partial.pt').read_bytes() == b'preserve'
    with pytest.raises(ValueError, match='within data/prepared'):
        rebuild.activate_alias(tmp_path / 'outside', fresh)
