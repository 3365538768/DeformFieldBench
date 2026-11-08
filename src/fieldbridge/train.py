"""DDP trainer with distributed validation and checkpoint-boundary control."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .config import load_config, ModelConfig
from .data import MaterialDataset, ShuffledOracleDataset, collate
from .evaluation import evaluate, move_batch
from .losses import compute_loss
from .local_model import build_model, initialize_rgb_baseline
from .runtime import atomic_json, paused, sha256, Plateau, checkpoint_weights, load_weights, forward_batch


def atomic_save(obj, path):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temporary); os.replace(temporary, path)


file_sha256 = sha256


def rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def broadcast(value, world):
    values = [value]
    if world > 1:
        dist.broadcast_object_list(values, src=0)
    return values[0]


def gather(value, world):
    values = [None] * world
    if world > 1:
        dist.all_gather_object(values, value)
    else:
        values = [value]
    return values


def run(config_path):
    cfg = load_config(config_path)
    world, rank, local_rank = (int(os.environ.get(k, default)) for k, default in [('WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0)])
    if cfg.train.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(f'cuda:{local_rank}' if cfg.train.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=30),
                                **({'device_id': device} if device.type == 'cuda' else {}))
    seed = cfg.train.seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path(cfg.train.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'last.pt').exists() and not cfg.train.resume:
            raise FileExistsError('Use a new output or explicitly resume')
    if world > 1:
        dist.barrier()
    def runtime(phase, **extra):
        if rank == 0:
            atomic_json(out / 'runtime.json', {'phase': phase, 'step': step, 'epoch': epoch + 1,
                        'updated_unix': time.time(), 'pause_requested': paused(out), 'field_end_epoch': field_end, **extra})
    dataset_class = MaterialDataset
    if cfg.model.architecture == 'direct_video':
        from .direct_data import FieldFreeDataset
        dataset_class = FieldFreeDataset
    train = dataset_class(cfg.data, 'train', cfg.model.field_size, cfg.model.field_mode)
    val = dataset_class(cfg.data, 'val', cfg.model.field_size, cfg.model.field_mode)
    if cfg.model.feedback == 'oracle_shuffled':
        train, val = ShuffledOracleDataset(train), ShuffledOracleDataset(val)
    if cfg.model.use_prior and not cfg.data.material_prior_file and not any(r.get('prior_parameters') for r in train.rows):
        raise ValueError('Independent priors missing')
    generator = torch.Generator().manual_seed(seed)
    sampler = DistributedSampler(train, shuffle=True, seed=cfg.train.seed) if world > 1 else None
    loader = DataLoader(train, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=cfg.train.workers, collate_fn=collate, pin_memory=device.type == 'cuda',
                        persistent_workers=bool(cfg.train.workers and cfg.data.prepared_cache_gib),
                        generator=generator, **({'prefetch_factor': 2} if cfg.train.workers else {}))
    raw_model = build_model(cfg.model).to(device)
    backbone_sha = broadcast(sha256(cfg.model.weights) if rank == 0 and cfg.model.weights else None, world)
    base_dataset = train.dataset if isinstance(train, ShuffledOracleDataset) else train
    if base_dataset.prepared:
        if base_dataset.prepared.get('backbone_checkpoint_sha256') != backbone_sha or base_dataset.prepared.get('hub_name') != cfg.model.hub_name:
            raise ValueError('Prepared feature backbone mismatch')
        if base_dataset.prepared.get('encoder_source_sha256') != sha256(Path(__file__).with_name('model.py')):
            raise ValueError('Prepared encoder implementation mismatch')
    head, backbone = [], []
    for name, p in raw_model.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith('frame_encoder.') else head).append(p)
    groups = [{'params': head, 'lr': cfg.train.lr}]
    if backbone:
        groups.append({'params': backbone, 'lr': cfg.train.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    start_epoch, step, epoch, pending = 0, 0, -1, False
    best_material, best_field = None, None
    field_end = cfg.train.field_epochs
    trackers = {'field': Plateau(cfg.train.field_plateau_patience, cfg.train.early_stop_relative_delta),
                'material': Plateau(cfg.train.early_stop_patience, cfg.train.early_stop_relative_delta)}
    manifest_hash = sha256(cfg.data.manifest)
    motion_hash = sha256(Path(cfg.data.motion_targets_dir)/'index.json') if cfg.data.motion_targets_dir else None
    rgb_hash=sha256(Path(cfg.data.rgb_cache_dir)/'index.json') if cfg.data.rgb_cache_dir else None
    overlays = {key: sha256(getattr(cfg.data,key)) if getattr(cfg.data,key) else None
                for key in ['material_prior_file','material_teacher_file']}
    feature_index_hash = sha256(Path(cfg.data.prepared_dir)/'index.json') if cfg.data.prepared_dir else None
    baseline_init_sha = sha256(cfg.train.initialize_rgb_baseline) if cfg.train.initialize_rgb_baseline else None
    if cfg.model.architecture == 'local_residual' and not (cfg.train.initialize_rgb_baseline or cfg.train.resume or cfg.train.initialize) and cfg.train.purpose != 'smoke':
        raise ValueError('Local residual research requires explicit trained baseline initialization')
    if cfg.train.initialize_rgb_baseline:
        baseline = torch.load(cfg.train.initialize_rgb_baseline, map_location='cpu', weights_only=False)
        initialize_rgb_baseline(raw_model, baseline, backbone_sha, manifest_hash, base_dataset.codec.stats)
    if cfg.train.initialize:
        load_weights(raw_model, torch.load(cfg.train.initialize, map_location='cpu', weights_only=False), backbone_sha)
    if cfg.train.resume:
        checkpoint = torch.load(cfg.train.resume, map_location='cpu', weights_only=False)
        if cfg.model.architecture == 'direct_video' and (checkpoint.get('material_overlay_sha256')!=overlays or checkpoint.get('feature_index_sha256')!=feature_index_hash):
            raise ValueError('Resume direct-video prior/teacher/feature provenance mismatch')
        if checkpoint['manifest_sha256'] != manifest_hash or asdict(ModelConfig(**checkpoint['config']['model'])) != cfg.dump()['model']:
            raise ValueError('Resume data/model mismatch')
        if checkpoint.get('motion_target_index_sha256')!=motion_hash:
            raise ValueError('Resume observed-motion supervision mismatch')
        if checkpoint.get('rgb_cache_index_sha256')!=rgb_hash:
            raise ValueError('Resume RGB cache mismatch')
        if checkpoint['world_size'] != world or not checkpoint.get('epoch_complete'):
            raise ValueError('Exact epoch resume requires same world size and complete epoch')
        if checkpoint['stats'] != base_dataset.codec.stats:
            raise ValueError('Resume codec mismatch')
        load_weights(raw_model, checkpoint, backbone_sha); optimizer.load_state_dict(checkpoint['optimizer'])
        baseline_init_sha = checkpoint.get('baseline_checkpoint_sha256')
        pending = checkpoint.get('validation_pending', False)
        start_epoch, step = checkpoint['epoch'] + (0 if pending else 1), checkpoint['step']
        best_material, best_field = checkpoint.get('best_material'), checkpoint.get('best_field')
        field_end = checkpoint.get('field_end_epoch', field_end)
        for name, state in checkpoint.get('plateau', {}).items():
            trackers[name].state = state
        restore_rng(checkpoint['rng_states'][rank]); generator.set_state(checkpoint['loader_states'][rank])
    model = DDP(raw_model, device_ids=[local_rank] if device.type == 'cuda' else None, broadcast_buffers=False) if world > 1 else raw_model
    source_hash = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob('*.py')):
        source_hash.update(f.name.encode()); source_hash.update(f.read_bytes())
    from .paths import path_map
    provenance = {'engine_version': 2, 'config': cfg.dump(), 'manifest_sha256': manifest_hash,
                  'motion_target_index_sha256': motion_hash, 'rgb_cache_index_sha256':rgb_hash,
                  'material_overlay_sha256': overlays, 'feature_index_sha256': feature_index_hash,
                  'runtime_path_map': path_map(),
                  'source_sha256': source_hash.hexdigest(), 'backbone_checkpoint_sha256': backbone_sha,
                  'stats': base_dataset.codec.stats, 'world_size': world, 'torch': torch.__version__,
                  'parameters': sum(p.numel() for p in raw_model.parameters()),
                  'trainable_parameters': sum(p.numel() for p in raw_model.parameters() if p.requires_grad),
                  'purpose': cfg.train.purpose, 'time_unix': time.time(), 'resume_from': cfg.train.resume,
                  'baseline_checkpoint_sha256': baseline_init_sha,
                  'privileged_diagnostic': cfg.model.feedback.startswith('oracle')}
    if rank == 0:
        if not (out / 'run.json').exists():
            atomic_json(out / 'run.json', provenance)
        atomic_json(out / f'session_{time.time_ns()}.json', provenance)
    if cfg.model.architecture == 'local_residual' and not cfg.train.resume:
        initial = evaluate(raw_model, val, device, out / 'val_initial', cfg.train.val_batch_size,
                           cfg.train.val_workers, cfg.train.val_bf16, distributed=world > 1)
        best_material = initial['standardized_mae']
        trackers['material'].state['best'] = best_material
        states, loader_states = gather(rng_state(), world), gather(generator.get_state(), world)
        if rank == 0:
            atomic_json(out / 'initial_evaluation.json', initial)
            weights, compact = checkpoint_weights(raw_model, cfg.train.compact_checkpoint)
            checkpoint = {'model': weights, 'compact_model': compact, 'optimizer': optimizer.state_dict(),
                          'motion_target_index_sha256': motion_hash, 'rgb_cache_index_sha256':rgb_hash,
                          'config': cfg.dump(), 'manifest_sha256': manifest_hash, 'stats': base_dataset.codec.stats,
                          'source_sha256': source_hash.hexdigest(), 'backbone_checkpoint_sha256': backbone_sha,
                          'baseline_checkpoint_sha256': baseline_init_sha,
                          'epoch': -1, 'epoch_complete': True, 'step': 0, 'validation_pending': False,
                          'best': best_material, 'best_material': best_material, 'best_field': None,
                          'field_end_epoch': field_end, 'world_size': world, 'rng_states': states,
                          'loader_states': loader_states, 'plateau': {k:v.state for k,v in trackers.items()}}
            atomic_save(checkpoint, out / 'best_material.pt')
            atomic_save(checkpoint, out / 'last.pt')
    amp = cfg.train.bf16 and device.type == 'cuda'
    reason = 'epochs_complete'
    for epoch in range(start_epoch, cfg.train.epochs):
        if broadcast(paused(out) if rank == 0 else None, world):
            runtime('paused', checkpoint=cfg.train.resume or str(out / 'last.pt')); reason = 'paused'; break
        if sampler:
            sampler.set_epoch(epoch)
        field_only = epoch < field_end
        phase = 'field_warmup' if field_only else 'joint_training'
        runtime(phase, field_end_epoch=field_end)
        epoch_start = time.perf_counter(); wait_seconds = 0.; complete = True
        model.train(); optimizer.zero_grad(set_to_none=True)
        if not pending:
            previous = time.perf_counter(); log_sums = None; log_count = 0
            for i, batch in enumerate(loader):
                wait_seconds += time.perf_counter() - previous
                batch = move_batch(batch, device)
                length = min(cfg.train.accumulation, len(loader) - (i // cfg.train.accumulation) * cfg.train.accumulation)
                sync = (i + 1) % cfg.train.accumulation == 0 or i + 1 == len(loader)
                with model.no_sync() if world > 1 and not sync else nullcontext():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                        prediction = forward_batch(model, batch, cfg.model.feedback)
                        loss, components = compute_loss(prediction, batch, cfg, field_only=field_only)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite loss')
                    (loss / length).backward()
                values = torch.stack([loss.detach(), *components.values()]).float()
                log_sums = values if log_sums is None else log_sums + values; log_count += 1
                if sync:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Non-finite gradient')
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                    if step <= 3 or step % cfg.train.log_interval == 0 or i + 1 == len(loader):
                        aggregate = log_sums / log_count
                        if world > 1:
                            dist.all_reduce(aggregate); aggregate /= world
                        if rank == 0:
                            numbers = aggregate.cpu().tolist()
                            record = {'epoch': epoch, 'step': step, 'loss_global': numbers[0],
                                      'loss_rank0': numbers[0], **dict(zip(components, numbers[1:])),
                                      'loss_scope': 'rank_mean_microbatch_mean', 'phase': phase,
                                      'data_wait_seconds_epoch': wait_seconds, 'updated_unix': time.time()}
                            with (out / 'train.jsonl').open('a') as f:
                                f.write(json.dumps(record) + '\n')
                            print(json.dumps(record), flush=True); runtime(phase, field_end_epoch=field_end)
                        log_sums = None; log_count = 0
                    if cfg.train.max_steps and step >= cfg.train.max_steps:
                        complete = i + 1 == len(loader); break
                previous = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        if world > 1:
            dist.barrier()
        train_seconds = time.perf_counter() - epoch_start
        states, loader_states = gather(rng_state(), world), gather(generator.get_state(), world)
        def save_checkpoint(validation_pending):
            if rank == 0:
                weights, compact = checkpoint_weights(raw_model, cfg.train.compact_checkpoint)
                checkpoint = {'model': weights, 'compact_model': compact, 'optimizer': optimizer.state_dict(),
                              'material_overlay_sha256': overlays, 'feature_index_sha256': feature_index_hash,
                              'motion_target_index_sha256': motion_hash, 'rgb_cache_index_sha256':rgb_hash,
                              'config': cfg.dump(), 'manifest_sha256': manifest_hash, 'stats': base_dataset.codec.stats,
                              'source_sha256': source_hash.hexdigest(), 'backbone_checkpoint_sha256': backbone_sha,
                              'baseline_checkpoint_sha256': baseline_init_sha,
                              'epoch': epoch, 'epoch_complete': complete, 'step': step,
                              'validation_pending': validation_pending, 'best': best_material,
                              'best_material': best_material, 'best_field': best_field, 'field_end_epoch': field_end,
                              'world_size': world, 'rng_states': states, 'loader_states': loader_states,
                              'plateau': {k: v.state for k, v in trackers.items()}}
                atomic_save(checkpoint, out / 'last.pt')
        save_checkpoint(True)
        runtime('validation', validation_epoch=epoch + 1, val_total=len(val))
        metrics = evaluate(raw_model, val, device, out / f'val_epoch_{epoch:04d}',
                           cfg.train.val_batch_size, cfg.train.val_workers, cfg.train.val_bf16,
                           distributed=world > 1,
                           progress=lambda value: atomic_json(out / f'val_progress_rank{rank}.json', {**value, 'epoch': epoch + 1}))
        score = metrics['standardized_mae']; field_score = metrics['field_absolute_error_encoded']
        improved_material = not field_only and score is not None and (best_material is None or score < best_material)
        improved_field = field_score is not None and (best_field is None or field_score < best_field)
        if improved_material:
            best_material = score
        if improved_field:
            best_field = field_score
        plateau = trackers['field' if field_only else 'material'].observe(
            field_score if field_only else score, 3 if field_only else cfg.train.early_stop_min_joint_epochs)
        if plateau and field_only:
            field_end = epoch + 1
        states, loader_states = gather(rng_state(), world), gather(generator.get_state(), world)
        save_checkpoint(False)
        if rank == 0:
            for improved, name in [(improved_material, 'best_material'), (improved_field, 'best_field')]:
                if improved:
                    shutil.copyfile(out / 'last.pt', out / f'{name}.pt.tmp'); os.replace(out / f'{name}.pt.tmp', out / f'{name}.pt')
            timing = {'epoch': epoch + 1, 'phase': phase, 'train_seconds': train_seconds,
                      'data_wait_seconds': wait_seconds, 'validation_seconds': metrics['validation_seconds'],
                      'cycle_seconds': time.perf_counter() - epoch_start, 'val_mae': score,
                      'field_mae': field_score, 'field_end_epoch': field_end,
                      'plateau': {k: v.state for k, v in trackers.items()}, 'coverage': metrics['coverage'],
                      'checkpoint_bytes': (out / 'last.pt').stat().st_size}
            with (out / 'timing.jsonl').open('a') as f:
                f.write(json.dumps(timing) + '\n')
            print(json.dumps(timing), flush=True)
        if world > 1:
            dist.barrier()
        pending = False
        if plateau and not field_only:
            reason = 'validation_material_plateau'; break
        if broadcast(paused(out) if rank == 0 else None, world):
            reason = 'paused'; runtime('paused', checkpoint=str(out / 'last.pt')); break
        if cfg.train.max_steps and step >= cfg.train.max_steps:
            reason = 'max_steps'; break
    if rank == 0:
        if reason != 'paused':
            status = {'status': 'early_stopped' if reason == 'validation_material_plateau' else 'software_smoke_passed' if cfg.train.purpose == 'smoke' else 'training_finished',
                      'reason': reason, 'steps': step, 'best_val_standardized_mae': best_material,
                      'best_field_mae': best_field, 'updated_unix': time.time(),
                      'privileged_diagnostic': cfg.model.feedback.startswith('oracle')}
            atomic_json(out / 'status.json', status); runtime('finished', **status)
        else:
            runtime('paused', checkpoint=str(out / 'last.pt'))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    run(p.parse_args().config)
