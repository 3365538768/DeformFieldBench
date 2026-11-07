import copy
import json
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset
from fieldbridge.local_model import build_model
from fieldbridge.evaluation import evaluate

torch.set_num_threads(1); dist.init_process_group('gloo')
cfg = load_config(sys.argv[1]); ds = MaterialDataset(cfg.data, 'val', cfg.model.field_size, cfg.model.field_mode)
torch.manual_seed(9); model = build_model(cfg.model)
for n in [5, 1]:
    subset = copy.copy(ds); subset.rows = ds.rows[:n]
    path = Path(sys.argv[2]) / str(n)
    result = evaluate(model, subset, 'cpu', path, batch_size=2, distributed=True)
    assert result['coverage']['unique'] == n
    if dist.get_rank() == 0:
        gold = evaluate(model, subset, 'cpu', batch_size=1)
        assert abs(gold['standardized_mae'] - result['standardized_mae']) < 1e-6
        records = [json.loads(s) for s in (path / 'predictions.jsonl').read_text().splitlines()]
        assert len({r['sample_id'] for r in records}) == n
    dist.barrier()
dist.destroy_process_group()
