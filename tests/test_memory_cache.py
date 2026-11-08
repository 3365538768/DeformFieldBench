import torch
from fieldbridge.memory_cache import TensorLRU
from fieldbridge.data import MaterialDataset
from fieldbridge.config import DataConfig


def test_lru_budget_aliases_and_oversized_sample():
    cache=TensorLRU(32);x=torch.ones(8)
    a=cache.get('a',lambda:{'x':x,'alias':x[:2]})
    assert cache.bytes==32
    assert cache.get('a',lambda:None) is a
    cache.get('b',lambda:{'x':torch.ones(16)})
    assert list(cache.items)==['a']
    cache.get('c',lambda:{'x':torch.zeros(8)})
    assert list(cache.items)==['c'] and cache.bytes==32


def test_augmented_cached_sample_does_not_corrupt_future_views(tmp_path,monkeypatch):
    sample={'sample_id':'a','valid':torch.ones(3,2,dtype=torch.bool),
            'field_mask':torch.ones(3,2,1,2,2)}
    torch.save(sample,tmp_path/'a.pt')
    ds=MaterialDataset.__new__(MaterialDataset)
    ds.rows=[{'sample_id':'a'}];ds.prepared={'files':{'a':'a.pt'}}
    ds.cfg=DataConfig(prepared_dir=str(tmp_path),view_dropout=.9)
    ds.split='train';ds.memory_cache=TensorLRU(4096)
    first=ds[0];first['valid'].zero_();first['field_mask'].zero_()
    monkeypatch.setattr(torch,'load',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('cache miss')))
    ds.cfg.view_dropout=0
    second=ds[0]
    assert second['valid'].all() and second['field_mask'].all()
