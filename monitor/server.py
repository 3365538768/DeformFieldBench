"""Local-only A800 dashboard. Python standard library; SSH uses existing config."""
import argparse
import base64
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
import re
import shlex
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Monitor:
    def __init__(self, host='a800', interval=15, remote_root='/mnt/afs_engine/linrui/rsi_real_test'):
        self.host, self.interval = host, interval
        if not remote_root.startswith('/') or '\n' in remote_root:
            raise ValueError('remote root must be an absolute Linux path')
        self.remote_root = remote_root.rstrip('/')
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.snapshot = None
        self.history = deque(maxlen=240)
        self.error = None
        self.attempt = None
        self.success = None
        self.collecting = False
        self.previous = None
        self.control_token = secrets.token_urlsafe(32)
        self.preview_cache = {}

    def command(self, command):
        return 'cd '+shlex.quote(self.remote_root)+' && FIELDBRIDGE_ROOT='+shlex.quote(self.remote_root)+' FIELDBRIDGE_HOST='+shlex.quote(self.host)+' '+command

    def simulation(self, identifier):
        if identifier and not re.fullmatch(r'(warp|uipc)_[0-9a-f]{20}',identifier):
            raise ValueError('案例 ID 不合法')
        with self.lock:
            cached=self.preview_cache.get(identifier)
            if cached and time.time()-cached[0] < (15 if not identifier else 3600):
                return cached[1]
        ssh = str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'System32/OpenSSH/ssh.exe') if os.name=='nt' else 'ssh'
        # Send only the case ID as data. All paths are allowlisted on the server.
        script=(HERE/'simulation_remote.py').read_text(encoding='utf-8')
        payload=script+'\n'
        payload=payload.replace("json.loads(sys.stdin.read())",repr({'case':identifier}))
        result=subprocess.run([ssh,'-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',
            '-o','ConnectTimeout=8',self.host,
            self.command('.venv/bin/python -')],input=payload,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=40,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        lines=[line for line in result.stdout.splitlines() if line.startswith('{')]
        if result.returncode or not lines:
            raise ValueError('无法读取已登记的仿真案例，请检查 SSH 或案例文件')
        value=json.loads(lines[-1])
        with self.lock:
            if len(self.preview_cache)>12:
                self.preview_cache.pop(next(iter(self.preview_cache)))
            self.preview_cache[identifier]=(time.time(),value)
        return value

    def rendered(self, identifier, asset):
        if (identifier or asset) and (not re.fullmatch(r'clip_[0-9a-f]{20}',identifier) or asset not in {'preview.gif','contact_sheet.png','final.png'}):
            raise ValueError('渲染案例或图像名称不合法')
        key='render:'+identifier+':'+asset
        with self.lock:
            cached=self.preview_cache.get(key)
            if cached and time.time()-cached[0]<(3600 if identifier else 15):return cached[1]
        ssh=str(Path(os.environ.get('WINDIR','C:/Windows'))/'System32/OpenSSH/ssh.exe') if os.name=='nt' else 'ssh'
        payload=(HERE/'render_remote.py').read_text(encoding='utf-8').replace(
            'json.loads(sys.stdin.read())',repr({'clip':identifier,'asset':asset}))
        result=subprocess.run([ssh,'-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=8',self.host,
            self.command('.venv/bin/python -')],input=payload,capture_output=True,
            text=True,encoding='utf-8',errors='replace',timeout=40,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        lines=[line for line in result.stdout.splitlines() if line.startswith('{')]
        if result.returncode or not lines:raise ValueError('无法读取已登记的渲染图像')
        value=json.loads(lines[-1])
        with self.lock:
            if len(self.preview_cache)>12:self.preview_cache.pop(next(iter(self.preview_cache)))
            self.preview_cache[key]=(time.time(),value)
        return value

    def state(self):
        with self.lock:
            age = time.time() - self.success if self.success else None
            return {'connection': 'connecting' if not self.snapshot and self.collecting else
                        'offline' if self.error else 'online' if self.snapshot else 'connecting',
                    'last_error': self.error, 'last_attempt_utc': self.attempt,
                    'last_success_unix': self.success, 'age_seconds': age,
                    'stale': age is None or age > self.interval * 3 or self.error is not None,
                    'collecting': self.collecting, 'interval_seconds': self.interval,
                    'snapshot': self.snapshot, 'history': list(self.history), 'control_token': self.control_token}

    def control(self, request):
        if not isinstance(request, dict) or set(request) != {'id', 'action'} or not isinstance(request['id'], str) or not isinstance(request['action'], str) or request['action'] not in {'pause', 'resume'}:
            raise ValueError('仅支持暂停和继续')
        state = self.state()
        managed = (state.get('snapshot') or {}).get('optimization') or {}
        if request['id'] != 'pipeline' and request['id'] not in {j['id'] for j in managed.get('jobs', [])}:
            raise ValueError('实验未注册，不能控制')
        if state['stale']:
            raise ValueError('采集已过期，恢复连接后再操作')
        if time.time() - managed.get('updated_unix', 0) > 25:
            raise ValueError('队列管理器心跳已过期，控制请求未发送')
        ssh = str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'System32/OpenSSH/ssh.exe') if os.name == 'nt' else 'ssh'
        result = subprocess.run([ssh, '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
            '-o', 'ConnectTimeout=8', self.host,
            self.command('.venv/bin/python scripts/control_jobs.py')],
            input=json.dumps(request), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=25,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        lines = [s for s in result.stdout.splitlines() if s.startswith('{')]
        value = json.loads(lines[-1]) if lines else {'ok': False, 'error': 'SSH 控制服务未返回结果'}
        if result.returncode or not value.get('ok'):
            raise ValueError(value.get('error', '控制请求失败'))
        self.wake.set()
        return value

    def run(self):
        while True:
            self.wake.clear()
            with self.lock:
                self.collecting = True
                self.attempt = utc_now()
            try:
                ssh = str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'System32/OpenSSH/ssh.exe') if os.name == 'nt' else 'ssh'
                args = [ssh, '-T', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes',
                        '-o', 'ConnectTimeout=8', '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=5',
                        '-o', 'ServerAliveCountMax=2', self.host,
                        self.command('python3 -')]
                result = subprocess.run(args, input=(HERE / 'collect_remote.py').read_text(encoding='utf-8'),
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=35,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                if result.returncode:
                    # Keep diagnostic errors, but never publish full SSH invocations or key paths.
                    raise RuntimeError(result.stderr.strip()[-1200:] or f'SSH exit {result.returncode}')
                lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
                if not lines:
                    raise ValueError('SSH returned no JSON snapshot')
                snap = json.loads(lines[-1])
                if snap.get('host') != self.host or snap.get('project') != self.remote_root or not isinstance(snap.get('experiments'), list):
                    raise ValueError('Unexpected snapshot format')
                rates = {'rx_bps': None, 'tx_bps': None}
                if self.previous:
                    delta = snap['server_unix'] - self.previous['server_unix']
                    common = set(snap['network']) & set(self.previous['network'])
                    if delta > 0 and common:
                        rates = {direction + '_bps': max(0, sum(snap['network'][n][direction + '_bytes'] -
                            self.previous['network'][n][direction + '_bytes'] for n in common) / delta)
                            for direction in ('rx', 'tx')}
                snap['network_rates'] = rates
                self.previous = snap
                points = {'time': snap['server_unix'], 'cpu': snap['cpu']['percent'],
                          'gpus': {str(g['index']): g['utilization'] for g in snap['gpus']}}
                with self.lock:
                    self.snapshot = snap
                    self.history.append(points)
                    self.success = time.time()
                    self.error = None
            except Exception as e:
                with self.lock:
                    self.error = f'{type(e).__name__}: {e}'
            finally:
                with self.lock:
                    self.collecting = False
            self.wake.wait(self.interval)


def serve(host='a800', port=8765, interval=15, remote_root='/mnt/afs_engine/linrui/rsi_real_test'):
    monitor = Monitor(host, interval, remote_root)

    class Handler(BaseHTTPRequestHandler):
        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}

        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith('GET /api/state'):
                return
            super().log_message(fmt, *args)

        def send(self, data, kind, status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'self'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.valid_host():
                self.send(b'Invalid host', 'text/plain', 403); return
            path = self.path.split('?', 1)[0]
            if path == '/api/state':
                self.send(json.dumps(monitor.state(), ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8')
            elif path == '/api/health':
                self.send(b'{"ok":true,"service":"a800-monitor"}', 'application/json')
            elif path == '/api/simulation':
                try:
                    query=parse_qs(urlsplit(self.path).query)
                    if set(query)-{'case'} or any(len(v)!=1 for v in query.values()):
                        raise ValueError('只支持一个案例 ID')
                    value=monitor.simulation(query.get('case',[''])[0])
                    self.send(json.dumps(value,allow_nan=False).encode(),'application/json')
                except (ValueError,subprocess.SubprocessError) as exc:
                    self.send(json.dumps({'status':'error','error':str(exc)},ensure_ascii=False).encode(),'application/json; charset=utf-8',409)
            elif path == '/api/render':
                try:
                    query=parse_qs(urlsplit(self.path).query)
                    if set(query)-{'clip','asset'} or any(len(v)!=1 for v in query.values()):raise ValueError('仅支持已登记的渲染案例与图像')
                    value=monitor.rendered(query.get('clip',[''])[0],query.get('asset',[''])[0])
                    if 'content_base64' in value:self.send(base64.b64decode(value['content_base64'],validate=True),value['mime'])
                    else:self.send(json.dumps(value,allow_nan=False).encode(),'application/json')
                except (ValueError,subprocess.SubprocessError) as exc:
                    self.send(json.dumps({'status':'error','error':str(exc)},ensure_ascii=False).encode(),'application/json; charset=utf-8',409)
            elif path in {'/', '/index.html', '/app.js', '/simulation.js', '/rendered.js', '/style.css'}:
                filename = 'index.html' if path == '/' else path.lstrip('/')
                kind = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[filename.rsplit('.', 1)[-1]]
                self.send((HERE / filename).read_bytes(), kind + '; charset=utf-8')
            else:
                self.send(b'Not found', 'text/plain', 404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if not self.valid_host() or origin not in {f'http://127.0.0.1:{port}', f'http://localhost:{port}'} or not secrets.compare_digest(self.headers.get('X-Control-Token', ''), monitor.control_token):
                self.send(b'{"ok":false,"error":"Invalid control origin or token"}', 'application/json', 403); return
            if self.path != '/api/control':
                self.send(b'Not found', 'text/plain', 404); return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 4096 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Invalid request body')
                result = monitor.control(json.loads(self.rfile.read(length)))
                self.send(json.dumps(result).encode(), 'application/json')
            except (ValueError, subprocess.SubprocessError) as e:
                self.send(json.dumps({'ok': False, 'error': str(e)}, ensure_ascii=False).encode(), 'application/json; charset=utf-8', 409)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=monitor.run, daemon=True, name='ssh-collector').start()
    (HERE / 'runtime.json').write_text(json.dumps({'pid': os.getpid(), 'url': f'http://127.0.0.1:{port}',
        'started_utc': utc_now(), 'interval_seconds': interval, 'ssh_host': host, 'remote_root': remote_root}, indent=2), encoding='utf-8')
    print(f'A800 monitor: http://127.0.0.1:{port} (SSH read every {interval}s)', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='a800')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--interval', type=int, default=15)
    p.add_argument('--remote-root', default='/mnt/afs_engine/linrui/rsi_real_test')
    a = p.parse_args()
    if not 5 <= a.interval <= 300:
        p.error('interval must be 5–300 seconds')
    serve(a.host, a.port, a.interval, a.remote_root)
