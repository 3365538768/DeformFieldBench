"""Fixed-checkpoint validation benchmark; never uses the test split."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset
from fieldbridge.evaluation import evaluate
from fieldbridge.model import MaterialFieldModel
from fieldbridge.runtime import atomic_json, load_weights, sha256

p = argparse.ArgumentParser()
for key in ['config', 'checkpoint', 'output']:
    p.add_argument('--' + key, required=True)
p.add_argument('--bf16', action='store_true'); p.add_argument('--workers', type=int, default=0)
p.add_argument('--batch', type=int, default=1); p.add_argument('--limit', type=int, default=64)
a = p.parse_args()
torch.set_num_threads(1)
world, rank, local = [int(os.environ.get(k, d)) for k, d in [('WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0)]]
device = torch.device(f'cuda:{local}'); torch.cuda.set_device(device)
if world > 1:
    dist.init_process_group('nccl', device_id=device)
cfg = load_config(a.config); ds = MaterialDataset(cfg.data, 'val', cfg.model.field_size, cfg.model.field_mode)
ds.rows = sorted(ds.rows, key=lambda r: r['sample_id'])[:a.limit]
model = MaterialFieldModel(cfg.model).to(device)
ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
load_weights(model, ckpt, sha256(cfg.model.weights) if ckpt.get('compact_model') else None)
del ckpt
out = Path(a.output)
result = evaluate(model, ds, device, out, a.batch, a.workers, a.bf16, world > 1,
                  progress=lambda v: atomic_json(out / f'progress_rank{rank}.json', v))
if rank == 0:
    print(json.dumps(result), flush=True)
if world > 1:
    dist.destroy_process_group()
