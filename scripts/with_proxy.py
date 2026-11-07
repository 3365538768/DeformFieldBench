"""Run one command with the host-private proxy; never log credentials.

Configuration lives outside the repository at ~/.config/fieldbridge/proxy.json,
mode 0600, with key ``proxy_url``. No credentials belong in configs or Git.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/with_proxy.py COMMAND [ARG ...]')
    path = Path(os.environ.get('FIELDBRIDGE_PROXY_CONFIG', Path.home() / '.config/fieldbridge/proxy.json'))
    if path.stat().st_mode & 0o077:
        raise SystemExit('Proxy config must have private permissions (chmod 600)')
    proxy = json.loads(path.read_text())['proxy_url']
    endpoint = urlsplit(proxy)
    if endpoint.scheme not in ('http', 'https') or not endpoint.hostname or not endpoint.port:
        raise SystemExit('Invalid proxy endpoint')
    env = os.environ.copy()
    for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
        env[key] = proxy
    bypass = set(filter(None, env.get('no_proxy', env.get('NO_PROXY', '')).split(',')))
    bypass.update(('localhost', '127.0.0.1', '::1'))
    env['no_proxy'] = env['NO_PROXY'] = ','.join(sorted(bypass))
    raise SystemExit(subprocess.run(sys.argv[1:], env=env).returncode)


if __name__ == '__main__':
    main()
