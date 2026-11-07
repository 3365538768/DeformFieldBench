"""Training-only field fit check; never reports its scores as validation quality."""
import argparse
import json
from pathlib import Path
import time
import torch
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate, model_inputs
from fieldbridge.evaluation import move_batch
from fieldbridge.local_model import build_model, initialize_rgb_baseline
from fieldbridge.losses import compute_loss
from fieldbridge.runtime import atomic_json, checkpoint_weights, load_weights, paused, Plateau, sha256
from fieldbridge.train import atomic_save, rng_state, restore_rng


@torch.no_grad()
def measure(model, samples, device, batch_size, amp):
    model.eval(); total = count = 0.; tp = fp = fn = 0
    for i in range(0, len(samples), batch_size):
        b = move_batch(collate(samples[i:i+batch_size]), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(**model_inputs(b))
        mask = b['field_mask']; gt = b['fields']; pred = out['fields'].float()
        total += float(((pred-gt).abs()*mask).sum()); count += float(mask.sum())
        p, g, m = pred[:, :, :, :1]>.5, gt[:, :, :, :1]>.5, mask[:, :, :, :1]>0
        tp += int((p & g & m).sum()); fp += int((p & ~g & m).sum()); fn += int((~p & g & m).sum())
    return {'train_encoded_field_mae': total/max(count, 1), 'train_contact_iou': tp/max(tp+fp+fn, 1),
            'train_contact_recall': tp/max(tp+fn, 1)}


def run(path):
    cfg = load_config(path); out = Path(cfg.train.output); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.train.seed); torch.set_num_threads(1)
    device = torch.device(cfg.train.device); amp = cfg.train.bf16 and device.type == 'cuda'
    ds = MaterialDataset(cfg.data, 'train', cfg.model.field_size, cfg.model.field_mode)
    indices = torch.linspace(0, len(ds)-1, min(64, len(ds))).round().long().tolist()
    samples = [ds[i] for i in indices]
    digest = sha256(cfg.data.manifest); backbone = sha256(cfg.model.weights) if cfg.model.weights else None
    if ds.prepared and (ds.prepared['backbone_checkpoint_sha256'] != backbone or
                        ds.prepared['encoder_source_sha256'] != sha256(Path(__file__).parents[1]/'src/fieldbridge/model.py')):
        raise ValueError('Prepared encoder provenance mismatch')
    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr)
    generator = torch.Generator().manual_seed(cfg.train.seed)
    tracker = Plateau(4, .002); start = step = 0
    initial = None; best = None
    if cfg.train.resume:
        ckpt = torch.load(cfg.train.resume, map_location='cpu', weights_only=False)
        if ckpt['manifest_sha256'] != digest or ckpt['sample_ids'] != [s['sample_id'] for s in samples]:
            raise ValueError('Training-only fit resume mismatch')
        if ckpt['config']['model'] != cfg.dump()['model'] or ckpt['stats'] != ds.codec.stats:
            raise ValueError('Fit model/statistics mismatch')
        load_weights(model, ckpt, backbone); optimizer.load_state_dict(ckpt['optimizer'])
        start, step, initial, best = ckpt['epoch']+1, ckpt['step'], ckpt['initial'], ckpt['best']
        tracker.state = ckpt['plateau']; generator.set_state(ckpt['generator'])
        restore_rng(ckpt['rng_state'])
    else:
        if (out/'last.pt').exists():
            raise FileExistsError('Explicit resume required')
        initialize_rgb_baseline(model, torch.load(cfg.train.initialize_rgb_baseline, map_location='cpu', weights_only=False),
                                backbone, digest, ds.codec.stats)
    if initial is None:
        initial = measure(model, samples, device, cfg.train.batch_size, amp)
    atomic_json(out/'run.json', {'config':cfg.dump(), 'purpose':'train_only_field_fit', 'samples':len(samples)})
    reason = 'epochs_complete'
    for epoch in range(start, cfg.train.epochs):
        if paused(out):
            atomic_json(out/'runtime.json', {'phase':'paused','epoch':epoch+1,'step':step,'updated_unix':time.time()}); return
        started=time.perf_counter(); model.train()
        for positions in torch.randperm(len(samples), generator=generator).split(cfg.train.batch_size):
            b = move_batch(collate([samples[i] for i in positions.tolist()]), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, components = compute_loss(model(**model_inputs(b)), b, cfg, field_only=True)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite fit loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            if not torch.isfinite(norm):raise FloatingPointError('Non-finite fit gradient')
            optimizer.step();step+=1
            atomic_json(out/'runtime.json', {'phase':'field_warmup','epoch':epoch+1,'step':step,'updated_unix':time.time(),'scope':'train_only_64'})
        metrics=measure(model,samples,device,cfg.train.batch_size,amp)
        improved=best is None or metrics['train_encoded_field_mae'] < best
        if improved:best=metrics['train_encoded_field_mae']
        stop=tracker.observe(metrics['train_encoded_field_mae'],5)
        weights,compact=checkpoint_weights(model,True)
        checkpoint={'model':weights,'compact_model':compact,'backbone_checkpoint_sha256':backbone,
                    'config':cfg.dump(),'manifest_sha256':digest,'stats':ds.codec.stats,
                    'sample_ids':[s['sample_id'] for s in samples],'optimizer':optimizer.state_dict(),
                    'epoch':epoch,'step':step,'initial':initial,'best':best,'plateau':tracker.state,
                    'generator':generator.get_state(),'rng_state':rng_state()}
        atomic_save(checkpoint,out/'last.pt')
        if improved:atomic_save(checkpoint,out/'best_field.pt')
        record={'epoch':epoch+1,'step':step,**metrics,'seconds':time.perf_counter()-started,'scope':'train_only_64'}
        with (out/'fit.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        atomic_json(out/'fit_progress.json',{'initial':initial,'latest':metrics,'epoch':epoch+1,'step':step})
        print(json.dumps(record),flush=True)
        if stop:reason='training_field_plateau';break
    result={'status':'training_finished','reason':reason,'samples':len(samples),'initial':initial,'final':metrics,
            'best_train_field_mae':best,'relative_field_improvement':1-best/initial['train_encoded_field_mae'],
            'scope':'training-only fit diagnostic, NOT generalization or SI physics','steps':step,'updated_unix':time.time()}
    atomic_json(out/'status.json',result);atomic_json(out/'runtime.json',{'phase':'finished','epoch':epoch+1,'step':step,'updated_unix':time.time()})
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
