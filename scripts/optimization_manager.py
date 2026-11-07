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
        effective = f"configs/runtime/{job['id']}.json"; atomic(ROOT / effective, cfg)
        job['config'] = effective
        args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={count}',
                '-m', 'fieldbridge.train', '--config', effective]
    elif count > 1:
        args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={count}', *job['args']]
    else:
        args = [sys.executable, *job['args']]
    log = ROOT / job['log']; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    children[job['id']] = process
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
    proceed = complete and o < c * .98 and d < c * .98 and d < g
    value = {'status': 'complete' if complete else 'incomplete', 'results': results,
             'resume_full_pair': bool(proceed), 'single_seed_diagnostic_only': True,
             'route': 'continue_full_pair' if proceed else 'revise_data_or_bridge_before_spending_full_budget',
             'note': 'Routing gate only; no significance or online-fitting claim.', 'updated_unix': time.time()}
    atomic(ROOT / 'docs/G1_DIAGNOSTIC_RESULTS.json', value)
    return value


def main():
    os.chdir(ROOT)
    lock = (ROOT / 'docs/.optimization_manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read('docs/OPTIMIZATION_PLAN.json')
    previous = read('docs/OPTIMIZATION_PROGRESS.json', {})
    jobs = previous.get('jobs', copy.deepcopy(plan['jobs']))
    tasks = previous.get('tasks', copy.deepcopy(plan['tasks']))
    for job in jobs:
        job.setdefault('source_config', next(j['config'] for j in plan['jobs'] if j['id'] == job['id']))
    children = {}; events = previous.get('events', [])
    def event(message):
        events.append({'time_unix': time.time(), 'message': message}); del events[:-40]
    event('实验队列已启动；暂停在检查点生效，停滞自动早停。')
    while True:
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
                cache = read('docs/G1_CACHE_STATUS.json', {})
                j['state'] = 'finished' if cache.get('status') == 'complete' else 'paused' if cache.get('status') == 'paused' else 'failed'
            else:
                j['state'] = 'finished' if code == 0 or code is None and (ROOT / f"runs/g0_benchmark/{j['id']}/summary.json").exists() else 'failed'
            j['exit_code'] = code; event(f"{j['id']}：{j['state']} {j.get('stop_reason') or ''}")
        benchmarks = [j for j in tasks if j['stage'] == 'G0']
        bench = read('docs/G0_BENCHMARK.json')
        if not bench and all(j['state'] == 'finished' for j in benchmarks):
            bench = benchmark_result(); event('G0 验证验收：' + bench['status'])
        cache_task = next(j for j in tasks if j['kind'] == 'cache')
        diagnostics = [j for j in jobs if j['stage'] == 'G1']
        diagnostic = read('docs/G1_DIAGNOSTIC_RESULTS.json')
        if not diagnostic and all(j['state'] in {'finished', 'failed'} for j in diagnostics):
            diagnostic = diagnostic_results(jobs); event('G1 反馈：' + diagnostic['route'])
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
                for j in diagnostics:
                    control = read(ROOT / j['output'] / 'control.json', {})
                    if j['state'] not in {'queued', 'paused'} or control.get('desired') == 'paused':
                        continue
                    free = [str(i) for i in range(4) if str(i) not in occupied and memory[i] < 1000]
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
            {'id': 'G2', 'title': '引擎标定与新动作响应', 'state': 'planned', 'detail': '待 G1 判断后实施；尚无实际响应结果'},
            {'id': 'G3', 'title': '度量场、真实数据与局部时序桥', 'state': 'planned', 'detail': '按上界诊断决定优先修数据或模型'},
            {'id': 'G4', 'title': '多种子与强方法对照', 'state': 'planned', 'detail': '冻结协议后进行，不能将诊断上界当作部署结果'}]
        for j in jobs:
            j['control'] = read(ROOT / j['output'] / 'control.json', {'desired': 'running'})
            j['runtime'] = read(ROOT / j['output'] / 'runtime.json', {})
        state = {'version': 2, 'manager_pid': os.getpid(), 'updated_unix': time.time(),
                 'desired': 'paused' if paused else 'running', 'stages': stages, 'jobs': jobs, 'tasks': tasks,
                 'drain': drain, 'cpu_tests_passed': test_passed, 'benchmark': bench,
                 'cache': read('docs/G1_CACHE_STATUS.json'), 'diagnostic': diagnostic, 'events': events,
                 'stop_policy': plan['stop_policy']}
        atomic('docs/OPTIMIZATION_PROGRESS.json', state)
        atomic('docs/ACTIVE_OPTIMIZATION_JOBS.json', {'purpose': 'v2 controlled diagnostic and continuation queue',
               'jobs': [j for j in jobs if 'pid' in j]})
        time.sleep(2)


if __name__ == '__main__':
    main()
