"""Read-only Linux snapshot. Executed over SSH using stdin; no GPU framework imports."""
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(os.environ.get('FIELDBRIDGE_ROOT', '/mnt/afs_engine/linrui/rsi_real_test'))


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def tail(path, limit=262144):
    try:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - limit))
            return f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return []


def numeric(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def json_lines(path, limit=65536):
    result = []
    for line in tail(path, limit):
        try:
            result.append(json.loads(line))
        except ValueError:
            pass
    return result


def query_gpu(fields, kind='gpu'):
    try:
        result = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + ','.join(fields),
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8, check=True)
        return [dict(zip(fields, [v.strip() for v in row])) for row in csv.reader(io.StringIO(result.stdout))]
    except (OSError, subprocess.SubprocessError) as e:
        return {'error': str(e)}


def cpu_ticks():
    values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    # guest values are already included in user/nice.
    return sum(values[:8]), values[3] + values[4]


def collect():
    now = time.time()
    first = cpu_ticks()
    time.sleep(.2)
    second = cpu_ticks()
    delta = second[0] - first[0]
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    cpu = {'percent': round(100 * (1 - (second[1] - first[1]) / delta), 1) if delta else None,
           'cores': os.cpu_count(), 'load': list(os.getloadavg())}
    memory = {'total': mem['MemTotal'], 'available': mem['MemAvailable'],
              'used': mem['MemTotal'] - mem['MemAvailable'], 'cached': mem.get('Cached', 0)}
    disks = []
    for mount in ('/', str(ROOT)):
        d = shutil.disk_usage(mount)
        disks.append({'mount': mount, 'total': d.total, 'used': d.used, 'free': d.free})
    network = {}
    for line in Path('/proc/net/dev').read_text().splitlines()[2:]:
        name, rest = line.split(':', 1)
        if name.strip() == 'lo':
            continue
        cols = rest.split()
        network[name.strip()] = {'rx_bytes': int(cols[0]), 'tx_bytes': int(cols[8])}
    raw = query_gpu(['index', 'name', 'uuid', 'utilization.gpu', 'memory.used', 'memory.total',
                     'temperature.gpu', 'power.draw', 'power.limit'])
    gpus, gpu_error = [], None
    if isinstance(raw, dict):
        gpu_error = raw['error']
    else:
        for row in raw:
            gpus.append({'index': int(row['index']), 'name': row['name'], 'uuid': row['uuid'],
                'utilization': numeric(row['utilization.gpu']), 'memory_used_mib': numeric(row['memory.used']),
                'memory_total_mib': numeric(row['memory.total']), 'temperature_c': numeric(row['temperature.gpu']),
                'power_w': numeric(row['power.draw']), 'power_limit_w': numeric(row['power.limit'])})
    processes = query_gpu(['gpu_uuid', 'pid', 'process_name', 'used_gpu_memory'], 'compute-apps')
    if isinstance(processes, dict):
        processes = []
    jobs = []
    optimization = read_json(ROOT / 'docs/OPTIMIZATION_PROGRESS.json', {})
    optimization['bridge_sensitivity'] = read_json(ROOT / 'docs/G1C_D_BRIDGE_SENSITIVITY_FP32.json')
    optimization['forced_bridge_sensitivity'] = read_json(ROOT / 'docs/G1D_D_BRIDGE_SENSITIVITY_FP32.json')
    optimization['latest_verification'] = read_json(ROOT / 'docs/NF_VERIFICATION.json') or read_json(ROOT / 'docs/D3_EXECUTION_VERIFICATION.json') or read_json(ROOT / 'docs/G3_STARTUP.json') or read_json(ROOT / 'docs/G2_STARTUP.json') or read_json(ROOT / 'docs/G1F_STARTUP.json') or read_json(ROOT / 'docs/G1E_STARTUP.json') or read_json(ROOT / 'docs/G1D_STARTUP.json')
    optimization['previous_bridge_health'] = read_json(ROOT / 'docs/G1E_PREVIOUS_BRIDGE_HEALTH.json')
    optimization['target_audit'] = read_json(ROOT / 'docs/G1E_TARGET_AUDIT.json')
    optimization['anchor_sensitivity'] = read_json(ROOT / 'docs/G1E_D_BRIDGE_SENSITIVITY_FP32.json')
    optimization['anchored_field_audit'] = read_json(ROOT / 'docs/G1E_D_FIELD_AUDIT.json')
    optimization['dice_field_audit'] = read_json(ROOT / 'docs/G1E_D_DICE_FIELD_AUDIT.json')
    optimization['dinov3_preparation'] = read_json(ROOT / 'docs/DINOV3_PREPARATION.json')
    optimization['strong_local_cache'] = read_json(ROOT / 'docs/G1F_LOCAL_CACHE.json')
    optimization['cache_rebuild'] = {name: read_json(ROOT / filename) for name, filename in
        [('dinov2', 'docs/CACHE_REBUILD_DINOV2.json'), ('dinov3', 'docs/CACHE_REBUILD_DINOV3.json')]}
    optimization['strong_field_audit'] = read_json(ROOT / 'docs/G1F_D_FIELD_AUDIT.json')
    optimization['strong_sensitivity'] = read_json(ROOT / 'docs/G1F_D_SENSITIVITY.json')
    optimization['cross_engine_calibration'] = read_json(ROOT / 'docs/G2_CROSS_ENGINE_CALIBRATION.json')
    optimization['cross_engine_results'] = read_json(ROOT / 'docs/G2_ADAPTER_RESULTS.json')
    optimization['engine_replay'] = read_json(ROOT / 'docs/G2_ENGINE_REPLAY.json')
    optimization['visibility_audit'] = read_json(ROOT / 'docs/G3_FORCE_VISIBILITY.json')
    mesh=read_json(ROOT / 'docs/G3_MESH_CONVERGENCE.json')
    optimization['mesh_quality']={k:v for k,v in mesh.items() if k!='rows'} if mesh else None
    rendered=read_json(ROOT / 'docs/G3_RENDER_QUALITY.json')
    optimization['render_quality']={k:v for k,v in rendered.items() if k!='clips'} if rendered else None
    contact=read_json(ROOT/'docs/G3_CONTACT_PREFLIGHT.json')
    optimization['contact_preflight']={k:v for k,v in contact.items() if k not in {'rows','clips'}} if contact else None
    contact_audit=read_json(ROOT/'docs/G3_CONTACT_NUMERICAL_AUDIT.json')
    optimization['contact_numerical_audit']={k:v for k,v in contact_audit.items() if k!='cases'} if contact_audit else None
    force_audit=read_json(ROOT/'docs/G3_CONTACT_FORCE_AUDIT.json')
    optimization['contact_force_audit']=force_audit
    optimization['contact_surface_audit']=read_json(ROOT/'docs/G3_CONTACT_SURFACE_AUDIT.json')
    optimization['contact_force_refinement']=read_json(ROOT/'docs/G3_CONTACT_FORCE_REFINEMENT.json')
    optimization['contact_regularization']=read_json(ROOT/'docs/G3_CONTACT_REGULARIZATION.json')
    optimization['external_motion']=read_json(ROOT/'docs/EXTERNAL_MOTION_V1.json')
    optimization['phystwin_eval_inputs']=read_json(ROOT/'docs/PHYSTWIN_EVAL_INPUTS.json')
    optimization['motion_dependence']=read_json(ROOT/'docs/M1_MOTION_DEPENDENCE.json')
    optimization['loader_efficiency']=read_json(ROOT/'docs/LOADER_EFFICIENCY.json')
    optimization['loader_workers']=read_json(ROOT/'docs/LOADER_WORKERS.json')
    optimization['motion_teacher']=read_json(ROOT/'docs/M2_TEACHER_AUDIT.json')
    motion_targets=read_json(ROOT/'docs/M2_OBSERVED_MOTION.json')
    optimization['observed_motion_targets']={k:v for k,v in motion_targets.items() if k!='rows'} if motion_targets else None
    optimization['observed_motion_dependence']=read_json(ROOT/'docs/M2_MOTION_DEPENDENCE.json')
    optimization['motion_bottlenecks']=read_json(ROOT/'docs/M2_BOTTLENECK_AUDIT.json')
    optimization['motion_deployment']=read_json(ROOT/'docs/M2_DEPLOYMENT_AUDIT.json')
    optimization['motion_gradients']=read_json(ROOT/'docs/M2_GRADIENT_AUDIT.json')
    optimization['material_pairs']=read_json(ROOT/'docs/MATERIAL_PAIR_AUDIT.json')
    optimization['single_video_prepare']=read_json(ROOT/'docs/D3_SINGLE_VIDEO_PREPARE.json')
    optimization['single_video_probe']=read_json(ROOT/'docs/D3_SINGLE_VIDEO_RESULTS.json')
    optimization['single_video_audit']=read_json(ROOT/'docs/D3_SINGLE_VIDEO_AUDIT.json')
    optimization['next_execution']=read_json(ROOT/'docs/D3_EXECUTION_ROADMAP.json')
    optimization['identifiability']=read_json(ROOT/'docs/I1_IDENTIFIABILITY.json')
    optimization['local_response']=read_json(ROOT/'docs/M3_LOCAL_RESPONSE_RESULTS.json')
    optimization['local_response_audit']=read_json(ROOT/'docs/M3_LOCAL_RESPONSE_AUDIT.json')
    optimization['pixel_prepare']=read_json(ROOT/'docs/B1_PIXEL_PREPARE.json')
    optimization['pixel_probe']=read_json(ROOT/'docs/B1_PIXEL_RESULTS.json')
    optimization['pixel_audit']=read_json(ROOT/'docs/B1_PIXEL_AUDIT.json')
    optimization['cache_precision']=read_json(ROOT/'docs/CACHE_PRECISION_PROBE.json')
    optimization['fieldfree_prepare']=read_json(ROOT/'docs/NF_STATIC_PREPARE.json')
    optimization['fieldfree_probe']=read_json(ROOT/'docs/NF_RESULTS.json')
    optimization['fieldfree_audit']=read_json(ROOT/'docs/NF_AUDIT.json')
    optimization['fieldfree_teacher']=read_json(ROOT/'docs/NF_TEACHER_FIXTURE.json')
    # Retries retain their job history, with one stage card for the latest attempt.
    optimization['stages']=list({s['id']:s for s in optimization.get('stages',[])}.values())
    efficiency=read_json(ROOT / 'docs/G3_INTERACTION_EFFICIENCY.json')
    optimization['interaction_efficiency']={k:v for k,v in efficiency.items() if k!='rows'} if efficiency else None
    optimization['data_realism_plan']=read_json(ROOT / 'docs/G3_DATA_V2_PLAN.json')
    optimization['cross_engine_training'] = read_json(ROOT / 'runs/g2_cross_engine_pilot/adapter/progress.json')
    optimization['engine_calibration'] = read_json(ROOT / 'docs/UIPC_CALIBRATION.json')
    optimization['beam_native'] = read_json(ROOT / 'docs/UIPC_BEAM_CALIBRATION.json')
    optimization['beam_corrected'] = read_json(ROOT / 'docs/UIPC_BEAM_CORRECTED_CALIBRATION.json')
    optimization['source_pilots'] = read_json(ROOT / 'docs/DATA_SOURCE_PILOTS.json')
    optimization['video_audit'] = read_json(ROOT / 'docs/EXTERNAL_VIDEO_AUDIT.json')
    optimization['engine_adapter'] = read_json(ROOT / 'docs/ENGINE_DOMAIN_ADAPTER_STATUS.json')
    for stage in optimization.get('stages', []):
        if stage['id'] == 'G2' and optimization['engine_calibration'] and not optimization.get('engine_progress'):
            stage.update(state='running', detail='libuipc CUDA 实测：力脉冲校准 6 场景通过；原生梁校准失败，物理系数修正与细网格后通过。复杂接触、混合材料及渲染仍待实现。')
        if stage['id'] == 'G3' and optimization['video_audit']:
            stage.update(state='running', detail='已有真实视频来源审计；受控双引擎场适配已完成，当前验证收敛网格及未放大渲染。视频模型训练与多引擎视频尚未整合。')
    for task in optimization.get('tasks', []):
        if task.get('kind') == 'benchmark':
            task['progress'] = [read_json(p, {}) for p in (ROOT / 'runs/g0_benchmark' / task['id']).glob('progress_rank*.json')]
    managed = {j['id']: j for j in optimization.get('jobs', [])}
    drained = {j['pid'] for j in optimization.get('drain', {}).get('jobs', []) if j['status'] == 'drained'}
    # Only explicit project-owned launch receipts; never inspect arbitrary user commands.
    for receipt in sorted((ROOT / 'docs').glob('ACTIVE*JOBS.json')):
        group = read_json(receipt, {})
        for job in group.get('jobs', []):
            if job.get('kind') in {'cache', 'cache_stage', 'engine_pipeline'}:
                continue
            config_name = job.get('config')
            args = job.get('args', [])
            if not config_name and '--config' in args:
                config_name = args[args.index('--config') + 1]
            if not config_name:
                continue
            config_path = ROOT / config_name
            cfg = read_json(config_path)
            if not cfg or 'train' not in cfg:
                continue
            if not config_path.resolve().is_relative_to(ROOT):
                continue
            out = (ROOT / cfg['train']['output']).resolve()
            if not out.is_relative_to(ROOT):
                continue
            if any(j['id'] == out.name for j in jobs):
                continue
            log = ROOT / job.get('log', '')
            train_path = out / 'train.jsonl'
            training = []
            for line in tail(train_path, 4 * 1024**2):
                try:
                    item = json.loads(line)
                    if 'step' in item:
                        training.append(item)
                except ValueError:
                    pass
            # Keep the curve bounded while preserving the most recent point.
            stride = max(1, math.ceil(len(training) / 600))
            curve = training[::stride]
            if training and (not curve or curve[-1] != training[-1]):
                curve.append(training[-1])
            latest = training[-1] if training else None
            validation = []
            for p in sorted(out.glob('val_epoch_*/summary.json')):
                summary = read_json(p)
                if summary:
                    validation.append({'epoch': int(p.parent.name.rsplit('_', 1)[-1]) + 1,
                        'mae': summary.get('standardized_mae'), 'field_mae': summary.get('field_absolute_error_encoded'),
                        'parameters': summary.get('parameters', {})})
            pid = job.get('pid')
            alive = False
            if pid:
                try:
                    cmd = Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode('utf-8', 'replace')
                    alive = 'torch.distributed.run' in cmd and str(config_name) in cmd
                except OSError:
                    pass
            final = read_json(out / 'status.json')
            status = 'finished' if final else 'running' if alive else 'stopped'
            last_lines = tail(log, 32768)[-16:]
            if not alive and not final and any('Traceback' in l or 'Error' in l for l in last_lines):
                status = 'failed'
            runtime = read_json(out / 'runtime.json', {})
            if not alive and runtime.get('phase') == 'paused':
                status = 'paused'
            if final and final.get('status') == 'early_stopped':
                status = 'early_stopped'
            if pid in drained and not alive:
                status = 'migrated'
            run = read_json(out / 'run.json', {})
            devices = str(job.get('devices', job.get('gpu_devices', '')))
            world = len([d for d in devices.split(',') if d.strip()]) or run.get('world_size', 1)
            counts = job.get('split_counts') or group.get('split_counts')
            if not counts:
                # The fixed pilot manifest has its own receipt.
                counts = read_json(ROOT / 'data/manifests/engineering_pilot_240.pilot.json', {}).get('split_counts', {})
            n_train = counts.get('train') if counts else None
            batch = cfg['train'].get('batch_size', 1)
            accum = cfg['train'].get('accumulation', 1)
            steps_epoch = math.ceil(math.ceil(math.ceil(n_train / world) / batch) / accum) if n_train else None
            step = latest['step'] if latest else 0
            epochs = cfg['train'].get('epochs', 0)
            warmup = cfg['train'].get('field_epochs', 0)
            epoch = latest.get('epoch', 0) + 1 if latest else 0
            last_change = train_path.stat().st_mtime if train_path.exists() else None
            phase = 'finished' if final else 'initializing' if not latest else 'field_warmup' if epoch <= warmup else 'joint_training'
            if latest and steps_epoch and step % steps_epoch == 0 and (not validation or validation[-1]['epoch'] < epoch) and alive:
                phase = 'validation_or_checkpoint'
            if runtime:
                phase = runtime.get('phase', phase); epoch = runtime.get('epoch', epoch)
                step = runtime.get('step', step)
                warmup = runtime.get('field_end_epoch', warmup)
            timing = json_lines(out / 'timing.jsonl')
            scores = [v['val_mae'] for v in timing if v.get('phase') == 'joint_training' and v.get('val_mae') is not None] if runtime else [v['mae'] for v in validation if v['mae'] is not None and v['epoch'] > warmup]
            initial = read_json(out / 'initial_evaluation.json', {})
            if initial.get('standardized_mae') is not None:
                scores.append(initial['standardized_mae'])
            start = job.get('launched_unix', run.get('time_unix'))
            elapsed = now - start if start else None
            eta = None
            if alive and steps_epoch and step >= steps_epoch and elapsed and step:
                eta = max(0, (epochs * steps_epoch - step) * elapsed / step)
            jobs.append({'id': out.name, 'name': job.get('name', out.name), 'status': status, 'phase': phase,
                'devices': devices, 'pid': pid, 'purpose': cfg['train'].get('purpose', 'research'), 'stage': job.get('stage'),
                'role': job.get('role'), 'architecture': cfg['model'].get('architecture', 'legacy'),
                'backbone': cfg['model'].get('hub_name', cfg['model'].get('backbone')),
                'feedback': cfg['model'].get('feedback'), 'contact_loss': cfg['train'].get('contact_loss', 'bce'),
                'seed': cfg['train'].get('seed'), 'frames': cfg['data'].get('frames'),
                'views': cfg['data'].get('max_views'), 'epochs': epochs, 'warmup_epochs': warmup,
                'epoch': epoch, 'step': step, 'steps_per_epoch': steps_epoch,
                'total_steps': epochs * steps_epoch if steps_epoch else None,
                'last_update_unix': last_change, 'started_unix': start, 'elapsed_seconds': elapsed,
                'eta_seconds': eta, 'latest': latest, 'train_curve': curve, 'validation': validation,
                'best_val_mae': min(scores) if scores else None, 'final': final,
                'log_tail': last_lines, 'output': cfg['train']['output'], 'config': str(config_path.relative_to(ROOT)),
                'split_counts': counts, 'runtime': runtime, 'managed': managed.get(out.name),
                'val_progress': [read_json(p, {}) for p in out.glob('val_progress_rank*.json')],
                'timing': timing,
                'privileged_diagnostic': cfg['model'].get('feedback', '').startswith('oracle')})
    jobs.sort(key=lambda j: (j['purpose'] != 'research', j['status'] != 'running', j['id']))
    audit = read_json(ROOT / 'docs/full_dataset_audit.json', {})
    return {'server_unix': now, 'server_time_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
        'host': os.environ.get('FIELDBRIDGE_HOST', 'a800'), 'project': str(ROOT),
        'migration': {**read_json(ROOT.parent/'migration/status.json', {}),
            'dataset_download': read_json(ROOT/'data/archives/dataset.download.json', {}),
            'weights_download': read_json(ROOT.parent/'migration/weights_download.json', {}),
            'acceptance': read_json(ROOT.parent/'migration/acceptance.json', {})},
        'uptime_seconds': float(Path('/proc/uptime').read_text().split()[0]),
        'cpu': cpu, 'memory': memory, 'disks': disks, 'network': network, 'gpus': gpus,
        'gpu_error': gpu_error, 'gpu_processes': processes, 'experiments': jobs, 'optimization': optimization,
        'data': {'phase': read_json(ROOT / 'docs/P0_PHASE_STATUS.json', {}),
            'audit': {k: audit.get(k) for k in ('status', 'audited_samples', 'objects', 'material_counts', 'action_counts')},
            'audit_failures': len(audit.get('failures', [])),
            'split': read_json(ROOT / 'data/manifests/deformfield_v1.split.json', {}).get('split_counts'),
            'geometry': {k: v for k, v in read_json(ROOT / 'docs/geometry_audit.json', {}).items() if k != 'records'},
            'profile': read_json(ROOT / 'docs/FORMAL_PROFILE_STATUS.json', {})}}


if __name__ == '__main__':
    print(json.dumps(collect(), ensure_ascii=False, allow_nan=False))
