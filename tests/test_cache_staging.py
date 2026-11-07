import importlib.util
import json
from pathlib import Path
import pytest
import sys

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Node-local A800 cache uses Linux symlinks')

spec = importlib.util.spec_from_file_location('staging', Path(__file__).parents[1] / 'scripts/stage_feature_cache.py')
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


def test_staging_verifies_and_repairs_local_copy_preserving_source(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'sample.pt').write_bytes(b'immutable original')
    (source / 'index.json').write_text(json.dumps({'files': {'sample': 'sample.pt'}}))
    target, link, receipt = [tmp_path / n for n in ('local', 'link', 'receipt.json')]
    staging.stage(source, target, link, receipt)
    assert link.resolve() == target
    (target / 'sample.pt').write_bytes(b'corrupt copy')
    staging.stage(source, target, link, receipt)
    assert (target / 'sample.pt').read_bytes() == b'immutable original'
    assert (source / 'sample.pt').read_bytes() == b'immutable original'
    assert json.loads(receipt.read_text())['files_verified_sha256'] == 1


def test_staging_refuses_external_index_paths(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'index.json').write_text(json.dumps({'files': {'sample': '../outside.pt'}}))
    with pytest.raises(ValueError, match='external path'):
        staging.stage(source, tmp_path / 'target', tmp_path / 'link', tmp_path / 'receipt.json')
