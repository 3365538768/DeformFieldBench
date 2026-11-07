"""Persistent G0/G1 queue and checkpoint-boundary training control for this project."""
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); os.replace(tmp, path)


def alive(job):
    try:
        command = Path(f"/proc/{job['pid']}/cmdline").read_text().replace('\0', ' ')
        token = job.get('config') if job['kind'] == 'train' else job['args'][0]
        return token in command and ('torch.distributed.run' in command or 'python' in command)
    except (OSError, KeyError):
        return False


def launch(job, children):
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=job['devices'])
    count = len(job['devices'].split(','))
    if job['kind'] == 'train':
        cfg = read(ROOT / job.get('source_config', job['config']))
        last = ROOT / job['output'] / 'last.pt'
        if last.exists():
            cfg['train']['resume'] = str(last.relative_to(ROOT))
            cfg['train']['initialize'] = ''
            cfg['train']['initialize_rgb_baseline'] = ''
        effective = f"configs/runtime/{job['id']}.json"; atomic(ROOT / effective, cfg)
        job['config'] = effective
        entry = [job['entrypoint']] if job.get('entrypoint') == 'scripts/overfit_fields.py' else ['-m', 'fieldbridge.train']
        args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={count}',
                *entry, '--config', effective]
    elif count > 1:
        args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={count}', *job['args']]
    else:
        args = [sys.executable, *job['args']]
    log = ROOT / job['log']; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    children[job['id']] = process
    job.pop('stop_reason', None)
    job.pop('exit_code', None)
    job.update(pid=process.pid, state='running', launched_unix=time.time(), args_launched=args)


def benchmark_result():
    base = ROOT / 'runs/g0_benchmark'
    names = ['gold_fp32', 'distributed_fp32', 'distributed_bf16']
    results = {n: read(base / n / 'summary.json') for n in names}
    records = {n: [json.loads(s) for s in (base / n / 'predictions.jsonl').read_text().splitlines()] for n in names}
    gold = {r['sample_id']: r for r in records[names[0]]}
    checks = {}
    for n in names[1:]:
        current = {r['sample_id']: r for r in records[n]}
        checks[n] = {'same_ids': set(current) == set(gold),
            'mae_delta': abs(results[n]['standardized_mae'] - results[names[0]]['standardized_mae']),
            'max_z_delta': max(abs(r['prediction_z'][i] - gold[s]['prediction_z'][i]) for s, r in current.items() for i in range(4)),
            'validation_speedup': results[names[0]]['validation_seconds'] / results[n]['validation_seconds']}
    fp, bf = checks[names[1]], checks[names[2]]
    passed = all(v['same_ids'] for v in checks.values()) and fp['mae_delta'] < 1e-5 and fp['max_z_delta'] < 1e-4 and bf['mae_delta'] < .01 and bf['max_z_delta'] < .05
    value = {'status': 'passed' if passed else 'failed', 'samples': len(gold), 'scope': 'fixed checkpoint, fixed val subset; forward+load+gather time, model setup excluded',
             'results': results, 'checks': checks, 'updated_unix': time.time()}
    atomic(ROOT / 'docs/G0_BENCHMARK.json', value)
    return value


def diagnostic_results(jobs):
    results = {}
    for j in jobs:
        if j['stage'] != 'G1':
            continue
        timing = ROOT / j['output'] / 'timing.jsonl'
        entries = [json.loads(s) for s in timing.read_text().splitlines()] if timing.exists() else []
        joint = [v for v in entries if v['phase'] == 'joint_training']
        best = min(joint, key=lambda r: r['val_mae']) if joint else None
        results[j['name']] = {'state': j['state'], 'best_material_mae': best['val_mae'] if best else None,
                             'best_epoch': best['epoch'] if best else None}
    c = results['C_aux_no_feedback']['best_material_mae']
    d = results['D_field_bridge']['best_material_mae']
    o = results['O_oracle_fields']['best_material_mae']
    g = results['G_feature_control']['best_material_mae']
    complete = all(v['best_material_mae'] is not None and v['state'] == 'finished' for v in results.values())
    b = results['B_rgb_only']['best_material_mae']
    proceed = complete and o < b * .98 and d < min(b, c, g) * .98
    value = {'status': 'complete' if complete else 'incomplete', 'results': results,
             'resume_full_pair': bool(proceed), 'single_seed_diagnostic_only': True,
             'route': 'continue_full_pair' if proceed else 'revise_data_or_bridge_before_spending_full_budget',
             'note': 'Routing gate only; no significance or online-fitting claim.', 'updated_unix': time.time()}
    atomic(ROOT / 'docs/G1_DIAGNOSTIC_RESULTS.json', value)
    return value


def followup_results(jobs, original):
    results = {}
    for job in jobs:
        if job['stage'] != 'G1b':
            continue
        path = ROOT / job['output'] / 'timing.jsonl'
        entries = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        joint = [v for v in entries if v['phase'] == 'joint_training']
        best = min(joint, key=lambda v: v['val_mae']) if joint else None
        results[job['name']] = {'state': job['state'], 'best_material_mae': best['val_mae'] if best else None,
                               'best_epoch': best['epoch'] if best else None, 'stop_reason': job.get('stop_reason')}
    terminal = bool(results) and all(v['state'] in {'finished', 'failed'} for v in results.values())
    complete = terminal and all(v['state'] == 'finished' and v['best_material_mae'] is not None for v in results.values())
    baseline = original['results']['B_rgb_only']['best_material_mae']
    observed = {k: v['best_material_mae'] for k, v in results.items() if v['best_material_mae'] is not None}
    best_name = min(observed, key=observed.get) if observed else None
    rgb = min(baseline, observed.get('B_rgb_repeat', baseline))
    comparisons = {}
    for suffix in ['weight01', 'weight001', 'weight01_detached', 'weight01_balanced']:
        bridge, control = 'D_' + suffix, 'C_' + suffix.replace('_detached', '')
        feature = 'G_weight01' if suffix in {'weight01', 'weight01_detached'} else None
        if bridge not in observed or control not in observed:
            continue
        best_control = min(rgb, observed[control], observed.get(feature, float('inf')))
        comparisons[bridge] = {'matching_aux_control': control, 'matching_feature_control': feature,
            'gain_over_rgb': 1 - observed[bridge] / rgb,
            'gain_over_aux_control': 1 - observed[bridge] / observed[control],
            'passes_2_percent_routing_threshold': observed[bridge] < best_control * .98,
            'feature_control_available': feature in observed}
    candidates = [v for v in comparisons.values() if v['passes_2_percent_routing_threshold']]
    next_route = ('repeat_independent_seeds' if any(v['feature_control_available'] for v in candidates)
                  else 'add_matched_feature_control' if candidates else 'repair_fields_and_local_temporal_bridge')
    value = {'status': 'complete' if complete else 'failed' if terminal else 'running', 'results': results,
             'original_rgb_mae': baseline, 'best_observed': best_name,
             'best_gain_over_original_rgb': 1 - observed[best_name] / baseline if best_name else None,
             'comparisons': comparisons,
             'route': next_route if complete else 'repair_failed_jobs' if terminal else 'collecting_evidence',
             'resume_full_pair': False, 'validation_tuning_only': True,
             'note': 'No automatic full-budget promotion: matched controls and independent seeds must be reviewed. Test split untouched.',
             'updated_unix': time.time()}
    atomic(ROOT / 'docs/G1B_RESULTS.json', value)
    return value


def merge_jobs(previous, planned):
    """Append new authorized jobs without resetting finished runs or their controls."""
    jobs = copy.deepcopy(previous)
    known = {job['id'] for job in jobs}
    jobs.extend(copy.deepcopy(job) for job in planned if job['id'] not in known)
    return jobs


def dependencies_finished(job, jobs):
    states = {j['id']: j['state'] for j in jobs}
    return all(states.get(identifier) == 'finished' for identifier in job.get('depends_on', []))


def available_gpu_devices(job, used, totals, occupied):
    """Shared GPUs are opt-in per job, with an explicit free-memory requirement."""
    minimum = job.get('resources', {}).get('minimum_free_mib')
    candidates = range(len(used)) if minimum is not None else range(min(4, len(used)))
    return [str(i) for i in candidates if str(i) not in occupied and
            (totals[i]-used[i] >= minimum if minimum is not None else used[i] < 1000)]


def strong_backbone_results(jobs):
    """Compare bridges against their own backbone's RGB estimator, never DINOv2."""
    trained = [j for j in jobs if j['kind'] == 'train']
    baseline = next((j for j in trained if j['name'] == 'B_rgb_only'), None)
    if baseline is None: return None
    timing = ROOT / baseline['output'] / 'timing.jsonl'
    records = [json.loads(s) for s in timing.read_text().splitlines()] if timing.exists() else []
    scores = [r['val_mae'] for r in records if r['phase'] == 'joint_training']
    if not scores:
        value = {'status':'running','results':{},'route':'prepare_cache_and_own_rgb_baseline',
                 'resume_full_pair':False,'updated_unix':time.time()}
        atomic(ROOT/'docs/G1F_RESULTS.json', value); return value
    value = local_bridge_results(trained, {'results':{'B_rgb_only':{'best_material_mae':min(scores)}}}, 'G1f')
    value['backbone'] = 'dinov3_vitl16'
    value['note'] = 'Matched DINOv3 RGB warm start; same 512/128 split and seed 17. Test untouched. No automatic full-budget promotion.'
    atomic(ROOT/'docs/G1F_RESULTS.json', value)
    return value


def local_bridge_results(jobs, original, stage='G1c'):
    results = {}
    for job in jobs:
        if job['stage'] != stage or job.get('role') == 'field_fit':
            continue
        path = ROOT / job['output'] / 'timing.jsonl'
        entries = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        joint = [v for v in entries if v['phase'] == 'joint_training']
        best = min(joint, key=lambda v: v['val_mae']) if joint else None
        initial = read(ROOT / job['output'] / 'initial_evaluation.json', {})
        initial_mae = initial.get('standardized_mae')
        selected = min([v for v in (initial_mae, best['val_mae'] if best else None) if v is not None], default=None)
        results[job['name']] = {'state': job['state'], 'best_material_mae': selected,
                               'initial_mae': initial_mae, 'best_trained_mae': best['val_mae'] if best else None,
                               'best_epoch': best['epoch'] if best and (initial_mae is None or best['val_mae'] < initial_mae) else 0,
                               'stop_reason': job.get('stop_reason')}
    members = [j for j in jobs if j['stage'] == stage]
    complete = bool(members) and all(j['state'] == 'finished' for j in members)
    failed = any(j['state'] == 'failed' for j in members)
    rgb = original['results']['B_rgb_only']['best_material_mae']
    d = results.get('D_local_fields', {}).get('best_material_mae')
    g = results.get('G_local_features', {}).get('best_material_mae')
    c = results.get('C_frozen_rgb', {}).get('best_material_mae', rgb)
    fit = next((j for j in members if j.get('role') == 'field_fit'), None)
    fit_result = read(ROOT / fit['output'] / 'status.json') if fit else None
    fit_progress = read(ROOT / fit['output'] / 'fit_progress.json') if fit else None
    passed = all(v is not None for v in (d,g,c)) and d < min(rgb,g,c)*.98
    value = {'status':'failed' if failed else 'complete' if complete else 'running', 'results':results,
             'field_fit':fit_result or fit_progress, 'original_rgb_mae':rgb,
             'route': 'repeat_independent_seeds' if complete and passed else 'review_field_fit_and_data_targets' if complete else 'collecting_evidence',
             'passes_2_percent_routing_threshold':bool(passed), 'resume_full_pair':False,
             'note':'Common frozen RGB warm start; validation tuning only. Epoch 0 selects the unchanged RGB estimator. Test untouched.',
             'updated_unix':time.time()}
    if stage == 'G1e':
        dd=results.get('D_dice',{}).get('best_material_mae')
        gg=results.get('G_dice',{}).get('best_material_mae')
        value['dice_matched_pair']={'D':dd,'G':gg,
            'passes_2_percent':bool(dd is not None and gg is not None and dd < min(rgb,gg)*.98),
            'scope':'Predeclared secondary contact-loss ablation; primary routing still uses balanced D/G.'}
    atomic(ROOT / f'docs/{stage.upper()}_RESULTS.json',value)
    return value


def main():
    os.chdir(ROOT)
    lock = (ROOT / 'docs/.optimization_manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read('docs/OPTIMIZATION_PLAN.json')
    previous = read('docs/OPTIMIZATION_PROGRESS.json', {})
    jobs = merge_jobs(previous.get('jobs', []), plan['jobs'])
    tasks = previous.get('tasks', copy.deepcopy(plan['tasks']))
    for job in jobs:
        job.setdefault('source_config', next(j['config'] for j in plan['jobs'] if j['id'] == job['id']))
    children = {}; events = previous.get('events', [])
    def event(message):
        events.append({'time_unix': time.time(), 'message': message}); del events[:-40]
    event('实验队列已启动；暂停在检查点生效，停滞自动早停。')
    while True:
        # New diagnostic stages can be appended while a solver is running.
        # Existing job state/control remains authoritative; never requeue history.
        latest_plan=read('docs/OPTIMIZATION_PLAN.json')
        known={j['id'] for j in jobs}
        additions=[j for j in latest_plan['jobs'] if j['id'] not in known]
        if additions:
            jobs=merge_jobs(jobs,additions)
            for j in jobs:
                j.setdefault('source_config',j.get('config'))
            event('已接入新阶段：'+', '.join(j['id'] for j in additions))
        global_control = read('docs/OPTIMIZATION_CONTROL.json', {})
        paused = global_control.get('desired') == 'paused'
        drain = read('docs/G0_DRAIN_STATUS.json', {})
        tests_log = Path('runs/phase_logs/g0_tests.log').read_text() if Path('runs/phase_logs/g0_tests.log').exists() else ''
        test_match = re.search(r'\b(\d+) passed\b', tests_log)
        test_passed = bool(test_match and int(test_match.group(1)) >= 14) and 'failed' not in tests_log.lower()
        for j in [*tasks, *jobs]:
            if j['state'] != 'running':
                continue
            process = children.get(j['id'])
            exited = process.poll() is not None if process else not alive(j)
            if not exited:
                continue
            code = process.returncode if process else None
            if j['kind'] == 'train':
                status = read(ROOT / j['output'] / 'status.json')
                runtime = read(ROOT / j['output'] / 'runtime.json', {})
                j['state'] = 'finished' if status else 'paused' if runtime.get('phase') == 'paused' else 'failed'
                j['stop_reason'] = status.get('reason') if status else runtime.get('phase')
            elif j['kind'] == 'cache':
                cache = read(j.get('status_file', 'docs/G1_CACHE_STATUS.json'), {})
                j['state'] = 'finished' if cache.get('status') == 'complete' else 'paused' if cache.get('status') == 'paused' else 'failed'
            elif j['kind'] == 'cache_stage':
                receipt = read(ROOT/j['status_file'], {})
                j['state'] = 'finished' if receipt.get('status')=='complete' else 'failed'
            elif j['kind'] == 'engine_pipeline':
                receipt = read(ROOT/j['output']/'status.json', {})
                j['state'] = 'finished' if receipt.get('status')=='complete' else 'failed'
                j['stop_reason'] = receipt.get('reason')
            else:
                j['state'] = 'finished' if code == 0 or code is None and (ROOT / f"runs/g0_benchmark/{j['id']}/summary.json").exists() else 'failed'
            j['exit_code'] = code; event(f"{j['id']}：{j['state']} {j.get('stop_reason') or ''}")
        benchmarks = [j for j in tasks if j['stage'] == 'G0']
        bench = read('docs/G0_BENCHMARK.json')
        if not bench and all(j['state'] == 'finished' for j in benchmarks):
            bench = benchmark_result(); event('G0 验证验收：' + bench['status'])
        cache_task = next(j for j in tasks if j['kind'] == 'cache')
        diagnostics = [j for j in jobs if j['stage'] == 'G1']
        followups = [j for j in jobs if j['stage'] == 'G1b']
        locals_ = [j for j in jobs if j['stage'] == 'G1c']
        forced = [j for j in jobs if j['stage'] == 'G1d']
        anchored = [j for j in jobs if j['stage'] == 'G1e']
        strong = [j for j in jobs if j['stage'] == 'G1f']
        strong_result = strong_backbone_results(strong) if strong else None
        engine_jobs = [j for j in jobs if j['kind']=='engine_pipeline' or j['stage'] in {'G2','G2r','G3v','G3c','G3r','G3t','G3p','G3q','G3f','G3s'}]
        engine_progress = read('docs/G2_ENGINE_PROGRESS.json', {})
        if strong_result and strong_result['status'] in {'complete','failed'} and (previous.get('strong_backbone') or {}).get('status') != strong_result['status']:
            event('G1f 反馈：' + strong_result['route'])
        diagnostic = read('docs/G1_DIAGNOSTIC_RESULTS.json')
        if not diagnostic and all(j['state'] in {'finished', 'failed'} for j in diagnostics):
            diagnostic = diagnostic_results(jobs); event('G1 反馈：' + diagnostic['route'])
        followup = followup_results(followups, diagnostic) if followups and diagnostic else None
        if followup and followup['status'] in {'complete', 'failed'} and (previous.get('followup') or {}).get('status') != followup['status']:
            event('G1b 反馈：' + followup['route'])
        local_bridge = local_bridge_results(locals_, diagnostic) if locals_ and diagnostic else None
        if local_bridge and local_bridge['status'] in {'complete','failed'} and (previous.get('local_bridge') or {}).get('status') != local_bridge['status']:
            event('G1c 反馈：' + local_bridge['route'])
        forced_bridge = local_bridge_results(forced, diagnostic, 'G1d') if forced and diagnostic else None
        if forced_bridge and forced_bridge['status'] in {'complete','failed'} and (previous.get('forced_bridge') or {}).get('status') != forced_bridge['status']:
            event('G1d 反馈：' + forced_bridge['route'])
        anchored_bridge = local_bridge_results(anchored, diagnostic, 'G1e') if anchored and diagnostic else None
        if anchored_bridge and anchored_bridge['status'] in {'complete','failed'} and (previous.get('anchored_bridge') or {}).get('status') != anchored_bridge['status']:
            event('G1e 反馈：' + anchored_bridge['route'])
        if not paused and test_passed and drain.get('status') == 'drained':
            busy_tasks = any(j['state'] == 'running' for j in tasks)
            if not busy_tasks and not any(j['state'] == 'failed' for j in tasks):
                task = next((j for j in benchmarks if j['state'] == 'queued'), None)
                if task is None and bench and bench['status'] == 'passed' and cache_task['state'] in {'queued', 'paused'}:
                    task = cache_task
                if task:
                    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                    if all(int(memory.splitlines()[int(d)].strip()) < 1000 for d in task['devices'].split(',')):
                        launch(task, children); event('开始 ' + task['id'])
            if bench and bench['status'] == 'passed' and cache_task['state'] == 'finished':
                occupied = {d for j in jobs if j['state'] == 'running' for d in j['devices'].split(',')}
                memory = [int(s.strip()) for s in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
                totals = [int(s.strip()) for s in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()]
                eligible = diagnostics + (followups if diagnostic and diagnostic['status'] == 'complete' else []) + (locals_ if followup and followup['status'] == 'complete' else []) + (forced if local_bridge and local_bridge['status'] == 'complete' else []) + (anchored if forced_bridge and forced_bridge['status'] == 'complete' else [])
                eligible += strong if anchored_bridge and anchored_bridge['status'] == 'complete' else []
                eligible += engine_jobs if strong_result and strong_result['status'] == 'complete' else []
                for j in eligible:
                    control = read(ROOT / j['output'] / 'control.json', {})
                    if j['state'] not in {'queued', 'paused'} or control.get('desired') == 'paused' or not dependencies_finished(j, jobs):
                        continue
                    if j.get('readiness_file') and not read(ROOT/j['readiness_file'], {}).get('runtime_ready'):
                        continue
                    if j.get('resources', {}).get('device') == 'cpu':
                        j['devices']=''; launch(j, children); event('开始 CPU 任务 ' + j['id']); continue
                    free = available_gpu_devices(j, memory, totals, occupied)
                    if free:
                        j['devices'] = free[0]; occupied.add(free[0]); launch(j, children); event('开始 ' + j['id'])
                for j in [j for j in jobs if j['stage'] == 'FULL']:
                    control = read(ROOT / j['output'] / 'control.json', {})
                    allowed = control.get('manual_start') or diagnostic and diagnostic.get('resume_full_pair')
                    if not allowed or j['state'] not in {'held_for_g1', 'paused', 'queued'} or control.get('desired') == 'paused':
                        continue
                    if all(d not in occupied and memory[int(d)] < 1000 for d in j['devices'].split(',')):
                        occupied.update(j['devices'].split(',')); launch(j, children); event('继续全量对照 ' + j['id'])
        stages = [
            {'id': 'G0', 'title': '效率与可靠性', 'state': 'passed' if bench and bench['status'] == 'passed' else 'failed' if bench or any(j['state'] == 'failed' for j in benchmarks) else 'running',
             'completed': sum(j['state'] == 'finished' for j in benchmarks) + int(test_passed) + int(drain.get('status') == 'drained'),
             'total': 5, 'detail': 'CPU 契约、旧实验检查点迁移、固定检查点 FP32/分布式/BF16 对照'},
            {'id': 'G1', 'title': '场信息上界与因果诊断', 'state': 'failed' if cache_task['state'] == 'failed' or any(j['state'] == 'failed' for j in diagnostics) else 'complete' if diagnostic else 'running' if cache_task['state'] != 'queued' else 'queued',
             'completed': sum(j['state'] == 'finished' for j in diagnostics), 'total': 6,
             'detail': '512 train / 128 val，物体隔离，单种子；含两组特权真场诊断'},
            {'id': 'G2', 'title': '双引擎标定与共享响应适配',
             'state': engine_progress.get('status', 'queued' if engine_jobs else 'planned'),
             'completed': engine_progress.get('completed', 0), 'total': engine_progress.get('total', 0),
             'detail': engine_progress.get('detail', '同夹具 Warp FEM / libuipc：先标定再生成受控响应矩阵；不混合原生参数。')},
            {'id': 'G3', 'title': '度量场、真实数据与局部时序桥', 'state': 'planned', 'detail': '按上界诊断决定优先修数据或模型'},
            {'id': 'G4', 'title': '多种子与强方法对照', 'state': 'planned', 'detail': '冻结协议后进行，不能将诊断上界当作部署结果'}]
        if followups:
            stages.insert(2, {'id': 'G1b', 'title': '负迁移与反馈梯度排查',
                'state': 'failed' if followup and followup['status'] == 'failed' else 'complete' if followup and followup['status'] == 'complete' else 'running' if any(j['state'] == 'running' for j in followups) else 'queued',
                'completed': sum(j['state'] == 'finished' for j in followups), 'total': len(followups),
                'detail': '固定数据/初始化：场权重 0.1、0.01，等容量控制、反馈梯度截断及接触平衡损失；含 RGB 复跑。完成后先审查再做多种子。'})
        if locals_:
            stages.insert(3, {'id':'G1c', 'title':'局部时序修正与场可拟合性',
                'state':local_bridge['status'], 'completed':sum(j['state']=='finished' for j in locals_), 'total':len(locals_),
                'detail':'冻结同一 RGB 基线，局部场/等容量特征修正、梯度截断、真场/置乱诊断；64 个训练样本只检验场可拟合性。'})
        if forced:
            stages.insert(4, {'id':'G1d','title':'移除修正头的直接 RGB 通路',
                'state':forced_bridge['status'],'completed':sum(j['state']=='finished' for j in forced),'total':len(forced),
                'detail':'仅改变修正头的直接 RGB 输入；预测场、同容量特征、真场与置乱真场匹配对照。基线仍冻结，测试集不参与。'})
        if anchored:
            stages.insert(5, {'id':'G1e','title':'场桥门归一化与零场参照',
                'state':anchored_bridge['status'],'completed':sum(j['state']=='finished' for j in anchored),'total':len(anchored),
                'detail':'修复门值整体塌缩和常量修正；D/G/O/S 主对照，加 D/G 稀疏接触 BCE+Dice 配对消融。测试集不参与。'})
        if strong:
            stages.insert(6, {'id':'G1f','title':'DINOv3 ViT-L 骨干与匹配场桥',
                'state':'failed' if any(j['state']=='failed' for j in strong) else 'complete' if all(j['state']=='finished' for j in strong) else 'running',
                'completed':sum(j['state']=='finished' for j in strong),'total':len(strong),
                'detail':'授权权重与特征一致性校验 → 独立缓存 → DINOv3 RGB 基线 → 冻结同一基线的 D/G/O/S；保持早停与暂停/继续。'})
        replay=read('docs/G2_ENGINE_REPLAY.json', {})
        if any(j['stage']=='G2r' for j in jobs):
            stages.insert(-2, {'id':'G2r','title':'预测参数回放与经典逆 FEM',
                'state':replay.get('status','queued'),'completed':replay.get('completed',0),'total':replay.get('total',180),
                'detail':replay.get('detail','相同完整 X/Y 位移观测；未见 Z 载荷的各引擎实际回放。')})
        visibility=read('docs/G3_FORCE_VISIBILITY.json', {})
        if any(j['stage']=='G3v' for j in jobs):
            stages.insert(-1, {'id':'G3v','title':'视频可观测性与载荷/网格审计',
                'state':visibility.get('status','queued'),'completed':visibility.get('completed',0),'total':visibility.get('total',20),
                'detail':visibility.get('detail','按真实 SI 形变筛查理想相机像素运动，不把可视化放大当视频监督。')})
        for stage, filename, title in [('G3c','G3_MESH_CONVERGENCE.json','细网格收敛与数据准入'),
                                       ('G3r','G3_RENDER_QUALITY.json','未放大视频与可见像素光流'),
                                       ('G3t','G3_INTERACTION_EFFICIENCY.json','有效动作采样、稳态早停与时间步'),
                                       ('G3p','G3_CONTACT_PREFLIGHT.json','不规则实体、粘接分区与三维工具接触预检'),
                                       ('G3q','G3_CONTACT_NUMERICAL_AUDIT.json','接触网格、时间步与均质极限诊断'),
                                       ('G3f','G3_CONTACT_FORCE_AUDIT.json','接触反力单位、能量与可辨识性诊断'),
                                       ('G3s','G3_CONTACT_SURFACE_AUDIT.json','完整保存帧的三角形/工具交叠与包含检查')]:
            if any(j['stage']==stage for j in jobs):
                quality=read('docs/'+filename,{})
                job=next(j for j in jobs if j['stage']==stage)
                stages.insert(-1,dict(id=stage,title=title,state=quality.get('status',job['state']),
                    completed=quality.get('completed',0),total=quality.get('total',0),
                    detail=quality.get('detail','等待前置质量检查；保留暂停/继续，不自动启动全量训练。')))
        for j in jobs:
            if j.get('progress_report'):
                quality=read(ROOT/j['progress_report'],{})
                stages.insert(-1,dict(id=j['stage'],title=j['name'],state=quality.get('status',j['state']),
                    completed=quality.get('completed',0),total=quality.get('total',0),
                    detail=quality.get('detail','有界诊断；未准入训练，不自动扩量。')))
            j['control'] = read(ROOT / j['output'] / 'control.json', {'desired': 'running'})
            j['runtime'] = read(ROOT / j['output'] / 'runtime.json', {})
        state = {'version': 2, 'manager_pid': os.getpid(), 'updated_unix': time.time(),
                 'desired': 'paused' if paused else 'running', 'stages': stages, 'jobs': jobs, 'tasks': tasks,
                 'drain': drain, 'cpu_tests_passed': test_passed, 'benchmark': bench,
                 'cache': read('docs/G1_CACHE_STATUS.json'), 'local_cache': read('docs/G1B_LOCAL_CACHE.json'),
                 'field_audit': read('docs/G1_PREDICTED_FIELD_AUDIT.json'),
                 'diagnostic': diagnostic, 'followup': followup, 'local_bridge': local_bridge,
                 'forced_bridge': forced_bridge, 'anchored_bridge': anchored_bridge, 'events': events,
                 'strong_backbone': strong_result, 'strong_cache': read('docs/G1F_CACHE_STATUS.json'),
                 'engine_progress': engine_progress,
                 'stop_policy': plan['stop_policy']}
        atomic('docs/OPTIMIZATION_PROGRESS.json', state)
        previous = state
        atomic('docs/ACTIVE_OPTIMIZATION_JOBS.json', {'purpose': 'v2 controlled diagnostic and continuation queue',
               'jobs': [j for j in jobs if 'pid' in j]})
        time.sleep(2)


if __name__ == '__main__':
    main()
