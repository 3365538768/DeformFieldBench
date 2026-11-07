"""Parallel raw-data prefetch must preserve order and skip existing samples."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('cache_builder',Path(__file__).parents[1]/'scripts/prepare_feature_cache.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_prefetch_matches_serial_and_does_not_load_completed_entries(tmp_path):
    class Dataset:
        rows=[{'sample_id':str(i)} for i in range(20)]
        def __getitem__(self,i):
            assert i!=3
            return {'value':i*i}
    (tmp_path/'three.pt').write_bytes(b'cached')
    meta={'files':{'3':'three.pt','5':'missing.pt'}}
    serial=list(builder.samples(Dataset(),meta,tmp_path,0))
    parallel=list(builder.samples(Dataset(),meta,tmp_path,4))
    assert serial==parallel
    assert [int(row['sample_id']) for row,sample in parallel]==[i for i in range(20) if i!=3]
    assert parallel[4][1]=={'value':25}  # Missing files are rebuilt, not silently skipped.
