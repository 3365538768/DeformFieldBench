import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from .data import collate, model_inputs, PARAMETERS
from .runtime import forward_batch, atomic_json


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def summarize(records, codec):
    result = {"samples": len(records), "filtered": False, "parameters": {}}
    if not records:
        raise ValueError("Empty evaluation")
    z = np.array([r["prediction_z"] for r in records])
    target = np.array([r["target_z"] for r in records])
    valid = np.array([r["target_mask"] for r in records], bool)
    std = np.array(codec.stats["std"])
    means = np.array(codec.stats["mean"])
    pred_raw = codec.decode(torch.tensor(z)).numpy()
    gt_raw = codec.decode(torch.tensor(target)).numpy()
    scores = []
    for j, name in enumerate(PARAMETERS):
        mask = valid[:, j]
        x, y = z[mask, j] * std[j] + means[j], target[mask, j] * std[j] + means[j]
        if not len(x):
            result["parameters"][name] = {"count": 0}
            continue
        ae = np.abs(x - y)
        raw_ae = np.abs(pred_raw[mask, j] - gt_raw[mask, j])
        denom = ((y - y.mean()) ** 2).sum()
        variance = np.exp(np.array([r["logvar"] for r in records])[mask, j]) * std[j] ** 2
        coverage = np.mean(ae <= 1.96 * np.sqrt(variance))
        scores.extend(np.abs(z[mask, j] - target[mask, j]).tolist())
        result["parameters"][name] = {
            "count": int(mask.sum()), "transformed_mae": float(ae.mean()),
            "transformed_median_ae": float(np.median(ae)), "transformed_p90_ae": float(np.quantile(ae, .9)),
            "raw_mae": float(raw_ae.mean()), "raw_median_ae": float(np.median(raw_ae)),
            "r2": float(1 - ((x-y)**2).sum()/denom) if denom > 0 else None,
            "pearson": float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>0 and y.std()>0 else None,
            "nominal_95_coverage_transformed": float(coverage)}
    result["standardized_mae"] = float(np.mean(scores)) if scores else None
    result["note"] = "Train-standardized ln1p/linear errors; raw outputs are constrained; not the legacy SPE. Coverage is empirical, not guaranteed."
    if all('mixture_logits' in r for r in records):
        from .posterior import posterior_summary
        result['posterior'] = posterior_summary(records)
        for j,name in enumerate(PARAMETERS):
            if result['parameters'][name]['count']:
                result['parameters'][name]['nominal_95_coverage_transformed'] = result['posterior']['per_parameter'][j]['coverage']
        result['note'] += ' Mixture coverage uses exact marginal CDF intervals; point-only runs do not train uncertainty.'
    return result


@torch.no_grad()
def evaluate(model, dataset, device, output=None, batch_size=1, workers=0, bf16=False,
             distributed=False, progress=None):
    model.eval()
    device = torch.device(device)
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed and dist.is_initialized() else (0, 1)
    shard = Subset(dataset, list(range(rank, len(dataset), world)))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    records = []
    field_num, field_den = 0., 0.
    loader = DataLoader(shard, batch_size=batch_size, shuffle=False, collate_fn=collate,
                        num_workers=workers, pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(12345),
                        **({'prefetch_factor': 2} if workers else {}))
    processed = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == 'cuda'):
            prediction = forward_batch(model, batch, model.cfg.feedback)
        if 'fields' in prediction:
            err = (prediction["fields"].float() - batch["fields"]).abs() * batch["field_mask"]
            field_num += err.sum().item()
            field_den += batch["field_mask"].sum().item()
        means = prediction['mean'].float().cpu().tolist()
        logvars = prediction['logvar'].float().cpu().tolist()
        laws = prediction['law_logits'].argmax(-1).cpu().tolist()
        targets, masks = batch['target'].cpu().tolist(), batch['target_mask'].cpu().tolist()
        gt_laws = batch['law'].cpu().tolist()
        for i, sid in enumerate(batch["sample_id"]):
            records.append({"sample_id": sid, "object_id": batch["object_id"][i], "source": batch["source"][i],
                            "action": batch["action"][i], "law_id": gt_laws[i], 'predicted_law_id': laws[i],
                            "prediction_z": means[i], "logvar": logvars[i],
                            "target_z": targets[i], "target_mask": masks[i]})
            if 'mixture_logits' in prediction:
                records[-1].update({k:prediction[k][i].float().cpu().tolist() for k in
                                   ['mixture_logits','mixture_means','mixture_logvars']})
        processed += len(batch['sample_id'])
        if progress:
            progress({'rank': rank, 'processed': processed, 'total': len(shard), 'updated_unix': time.time()})
    if world > 1:
        shards = [None] * world
        dist.all_gather_object(shards, records)
        records = [r for part in shards for r in part]
        totals = torch.tensor([field_num, field_den], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        field_num, field_den = totals.cpu().tolist()
    expected = {r['sample_id'] for r in dataset.rows}
    if len(records) != len(expected) or {r['sample_id'] for r in records} != expected:
        raise ValueError('Validation coverage mismatch: duplicated or missing sample IDs')
    records.sort(key=lambda r: r['sample_id'])
    result = summarize(records, dataset.codec)
    result['validation_seconds'] = time.perf_counter() - started
    result['coverage'] = {'expected': len(expected), 'unique': len(records), 'duplicates': 0, 'world_size': world}
    result['precision'] = 'bf16' if bf16 and device.type == 'cuda' else 'fp32'
    result['privileged_diagnostic'] = model.cfg.feedback.startswith('oracle')
    result["field_absolute_error_encoded"] = field_num / field_den if field_den else None
    result["by_source"] = {s: summarize([r for r in records if r["source"] == s], dataset.codec)
                           for s in sorted({r["source"] for r in records})}
    result["by_action"] = {s: summarize([r for r in records if r["action"] == s], dataset.codec)
                           for s in sorted({r["action"] for r in records})}
    result["by_law"] = {str(s): summarize([r for r in records if r["law_id"] == s], dataset.codec)
                        for s in sorted({r["law_id"] for r in records})}
    if output and rank == 0:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'summary.json', result)
        (output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return result
