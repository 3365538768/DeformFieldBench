"""Bounded single-GPU profile, not a training-quality experiment."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate, model_inputs
from fieldbridge.evaluation import move_batch
from fieldbridge.losses import compute_loss
from fieldbridge.local_model import build_model


def run(config, output):
    cfg=load_config(config)
    ds=MaterialDataset(cfg.data,"train",cfg.model.field_size,cfg.model.field_mode)
    torch.manual_seed(17)
    model=build_model(cfg.model).cuda().train()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg.train.lr)
    steps=[]
    for i in range(6):
        cpu_start=time.perf_counter()
        batch=collate([ds[i % len(ds)]])
        load_seconds=time.perf_counter()-cpu_start
        batch=move_batch(batch,"cuda")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=cfg.train.bf16):
            prediction=model(**model_inputs(batch))
            loss,_=compute_loss(prediction,batch,cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite profile loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,cfg.train.clip_grad)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite profile gradient")
        optimizer.step();torch.cuda.synchronize()
        record={"iteration":i,"gpu_step_seconds":time.perf_counter()-start,
                "cpu_read_resize_seconds":load_seconds,"loss":float(loss)}
        steps.append(record);print(json.dumps(record),flush=True)
    result={"status":"profile_passed","config":cfg.dump(),"iterations":steps,
        "median_gpu_step_seconds_after_warmup":statistics.median(x["gpu_step_seconds"] for x in steps[1:]),
        "max_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),
        "max_gpu_reserved_bytes":torch.cuda.max_memory_reserved(),
        "parameters":sum(p.numel() for p in model.parameters()),
        "trainable_parameters":sum(p.numel() for p in parameters),
        "note":"Six steps on training samples only; warmup excluded from median. No model-quality conclusion or test evaluation."}
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(a.config,a.output)
