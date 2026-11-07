"""Stage an existing immutable feature cache onto node-local storage; verify all files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from fieldbridge.runtime import atomic_json, sha256


def stage(source, destination, link, receipt):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    link = Path(link).absolute()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Source and destination must be separate directories')
    if link.exists() and link.resolve() != destination:
        raise ValueError('Refusing to replace an existing cache path')
    index = json.loads((source / 'index.json').read_text())
    names = sorted(set(index['files'].values()))
    for name in names:
        if not (source / name).resolve().is_relative_to(source) or Path(name).is_absolute():
            raise ValueError('Cache index contains an external path')
    total = sum((source / name).stat().st_size for name in names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and shutil.disk_usage(destination.parent).free < total * 1.1:
        raise RuntimeError('Insufficient node-local space')
    destination.mkdir(exist_ok=True)
    if (destination / 'index.json').exists() and sha256(destination / 'index.json') != sha256(source / 'index.json'):
        raise ValueError('Existing destination belongs to another cache')
    started = time.perf_counter()
    def copy_verify(name):
        original, target = source / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256(original)
        if not target.exists() or sha256(target) != expected:
            temporary = target.with_suffix(target.suffix + '.copying')
            shutil.copy2(original, temporary)
            if sha256(temporary) != expected:
                raise IOError('Cache copy failed SHA256 verification')
            temporary.replace(target)
        return name, expected
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(copy_verify, names))
    shutil.copy2(source / 'index.json', destination / 'index.json')
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        if link.is_symlink():
            raise ValueError('Refusing to replace a broken link')
        link.symlink_to(destination, target_is_directory=True)
    result = {'status': 'complete', 'source': str(source), 'destination': str(destination), 'link': str(link),
              'files_verified_sha256': len(verified), 'bytes': total, 'seconds': time.perf_counter()-started,
              'index_sha256': sha256(source / 'index.json'),
              'file_hashes_digest': hashlib.sha256(json.dumps(verified, separators=(',', ':')).encode()).hexdigest(),
              'note': 'Byte-identical cache relocation only. Re-run after node-local cache eviction; original AFS cache preserved.'}
    atomic_json(receipt, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('source', 'destination', 'link', 'receipt'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    stage(args.source, args.destination, args.link, args.receipt)
