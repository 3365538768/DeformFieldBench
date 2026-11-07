"""Allowlisted project control, accepting JSON on stdin from the local dashboard."""
import json
import os
from pathlib import Path
import sys
import time
import fcntl

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value)); os.replace(tmp, path)


def control(request):
    if set(request) != {'id', 'action'} or request['action'] not in {'pause', 'resume'}:
        raise ValueError('Only pause/resume is supported')
    plan = json.loads((ROOT / 'docs/OPTIMIZATION_PLAN.json').read_text())
    if request['id'] == 'pipeline':
        path = ROOT / 'docs/OPTIMIZATION_CONTROL.json'
    else:
        job = next((j for j in plan['jobs'] if j['id'] == request['id']), None)
        if job is None:
            raise ValueError('Unknown managed experiment')
        out = (ROOT / job['output']).resolve()
        if not out.is_relative_to(ROOT / 'runs'):
            raise ValueError('Invalid registered output')
        if (out / 'status.json').exists():
            raise ValueError('Completed/early-stopped runs require a new experiment, not resume')
        path = out / 'control.json'
    with (ROOT / 'docs/.control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = {'desired': 'paused' if request['action'] == 'pause' else 'running',
                 'updated_unix': time.time(), 'source': 'local_dashboard',
                 'manual_start': request['id'] != 'pipeline' and request['action'] == 'resume'}
        atomic(path, value)
        with (ROOT / 'docs/control_events.jsonl').open('a') as f:
            f.write(json.dumps({**request, **value}) + '\n')
    return {'ok': True, **request, **value,
            'message': 'Pause takes effect at a complete checkpoint; resume retains optimizer/RNG state.'}


if __name__ == '__main__':
    try:
        print(json.dumps(control(json.loads(sys.stdin.read()))))
    except Exception as e:
        print(json.dumps({'ok': False, 'error': str(e)})); sys.exit(1)
