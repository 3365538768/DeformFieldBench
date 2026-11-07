"""Migrate only registered project jobs after their next atomic epoch checkpoint."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def matches(job):
    try:
        cmd = Path(f"/proc/{job['pid']}/cmdline").read_bytes().decode()
        return 'torch.distributed.run' in cmd and job['config'] in cmd and os.getpgid(job['pid']) == job['pid']
    except (OSError, ProcessLookupError):
        return False


def main():
    path = ROOT / 'docs/G0_DRAIN_STATUS.json'
    if path.exists():
        raise FileExistsError('A migration is already registered')
    receipt = json.loads((ROOT / 'docs/ACTIVE_FULL_CD_JOBS.json').read_text())
    state = {'started_unix': time.time(), 'status': 'waiting_checkpoint', 'jobs': []}
    for j in receipt['jobs']:
        c = json.loads((ROOT / j['config']).read_text())
        ck = ROOT / c['train']['output'] / 'last.pt'
        state['jobs'].append({**j, 'checkpoint': str(ck.relative_to(ROOT)),
                              'original_mtime_ns': ck.stat().st_mtime_ns if ck.exists() else 0,
                              'status': 'waiting_checkpoint'})
    write(path, state)
    while any(j['status'] == 'waiting_checkpoint' for j in state['jobs']):
        for j in state['jobs']:
            if j['status'] != 'waiting_checkpoint':
                continue
            ck = ROOT / j['checkpoint']
            if not matches(j):
                j['status'] = 'exited'; continue
            if ck.exists() and ck.stat().st_mtime_ns > j['original_mtime_ns']:
                # best.pt is also atomic; allow its write and end-of-epoch barrier to finish.
                if any(ck.parent.glob('*.pt.tmp')) or time.time() - ck.stat().st_mtime < 3:
                    continue
                if matches(j):
                    os.killpg(j['pid'], signal.SIGTERM)
                    j.update(status='drained', stopped_unix=time.time(), checkpoint_bytes=ck.stat().st_size)
        state['updated_unix'] = time.time()
        if time.time() - state['started_unix'] > 3600:
            state['status'] = 'needs_attention'; write(path, state)
            raise TimeoutError('No safe checkpoint within one hour; no forced termination')
        write(path, state)
        time.sleep(.5)
    state['status'] = 'drained'; state['updated_unix'] = time.time(); write(path, state)


if __name__ == '__main__':
    main()
