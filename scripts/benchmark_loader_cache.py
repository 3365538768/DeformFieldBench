"""Bounded real-cache CPU loading benchmark; no model or training changes."""
import os
import argparse
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, Subset
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate
from engine_fixture import atomic, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);torch.set_num_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile-workers',action='store_true')
    parser.add_argument('--output',default='runs/loader_efficiency')
    parser.add_argument('--report',default='docs/LOADER_EFFICIENCY.json')
    args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    modes=[(0,0.),(2,0.),(4,0.),(0,4.),(2,4.)] if args.profile_workers else [(2,0.),(2,4.)]
    report=dict(status='running',completed=0,total=len(modes),rows=[],
        detail='相同 64 条缓存、固定顺序两遍读取；区分磁盘、进程间复制和有界 RAM 缓存。')
    try:
        for workers,budget in modes:
            cfg=load_config('configs/runtime/m1_B_motion_rgb_seed17.json')
            cfg.data.prepared_cache_gib=budget
            ds=MaterialDataset(cfg.data,'train',cfg.model.field_size,cfg.model.field_mode)
            ds.cfg.view_dropout=0
            loader=DataLoader(Subset(ds,range(64)),batch_size=4,num_workers=workers,collate_fn=collate,
                              persistent_workers=bool(workers))
            passes=[];digests=[]
            for epoch in range(2):
                wait_if_paused(output);begin=time.perf_counter();ids=[];checksum=0.
                for b in loader:
                    ids.extend(b['sample_id'])
                    features=b['features']
                    checksum+=float((features[:,:,:,0,0,0] if args.profile_workers else features.double()).sum())
                if len(set(ids))!=64:raise ValueError('Benchmark coverage mismatch')
                digests.append((ids,checksum));passes.append(time.perf_counter()-begin)
            if digests[0]!=digests[1]:raise ValueError('Cache changed tensors or sample order')
            report['rows'].append(dict(budget_gib_per_worker=budget,workers=workers,passes_seconds=passes,
                same_probe_and_order=True,samples=64,feature_sum=digests[0][1],
                checksum_scope='one feature element per view/frame' if args.profile_workers else 'all feature elements float64'))
            del loader,ds
            report.update(completed=len(report['rows']),updated_unix=time.time())
            atomic(ROOT/args.report,report)
        best=min(report['rows'],key=lambda r:r['passes_seconds'][1])
        report.update(status='complete',best_bounded_configuration=best,
            warm_pass_speedup=report['rows'][0]['passes_seconds'][1]/best['passes_seconds'][1],
            scope='Loading/collation/checksum only, CPU, bounded 64 samples; not end-to-end training speedup. Budget counts tensor storage per worker, excludes Python/IPC/OS page cache.')
        if len({r['feature_sum'] for r in report['rows']})!=1:raise ValueError('Different benchmark features')
        atomic(ROOT/args.report,report)
        atomic(output/'status.json',dict(status='complete',reason='bounded_loader_benchmark_complete'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/args.report,report)
        atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
