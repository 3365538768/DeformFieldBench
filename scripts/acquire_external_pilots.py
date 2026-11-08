"""Bounded public-data pilots with pinned revisions and local SHA256 receipts.

No material-parameter labels are invented. Deform360 supplies real RGB/timing
examples; PhysTwin ZIP ranges inspect available schemas before a full download.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

from fetch_sample_subset import HTTPRange


def configure_proxy(project_root):
    """Standard environment first; private deployment config is optional."""
    proxy=os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    private=Path(project_root).parent/'.private/fieldbridge/proxy.json'
    if not proxy and private.exists():proxy=json.loads(private.read_text())['proxy_url']
    if proxy:
        urllib.request.install_opener(urllib.request.build_opener(
            urllib.request.ProxyHandler({'https':proxy,'http':proxy})))


def get_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def deform360(root, object_ids, before_file=None):
    repo = 'brownu/deform360'
    info = get_json(f'https://huggingface.co/api/datasets/{repo}')
    revision = info['sha']
    candidates = sorted(s['rfilename'] for s in info['siblings'] if s['rfilename'].endswith('.mp4'))
    selected=[]
    for object_id in object_ids:
        matches=[name for name in candidates if name.split('/')[1]==object_id]
        if not matches:raise ValueError('Requested object is not published: '+object_id)
        selected.append(matches[0])
    existing = {s['rfilename'] for s in info['siblings']}
    selected += [n[:-4]+'.txt' for n in selected if n[:-4]+'.txt' in existing]
    receipts, budget = [], 256 * 1024**2
    for name in selected:
        if before_file:before_file(name)
        folder = name.rsplit('/', 1)[0]
        entries = get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{urllib.parse.quote(folder)}')
        entry = next(e for e in entries if e['path'] == name)
        size = entry['size']
        if size > 96 * 1024**2 or size > budget:
            receipts.append({'path': name, 'status': 'skipped_budget', 'bytes': size})
            continue
        url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(name)}'
        destination = root / 'deform360' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix+'.partial')
        expected_sha = (entry.get('lfs') or {}).get('oid')
        if destination.exists() and destination.stat().st_size==size and expected_sha and hashlib.sha256(destination.read_bytes()).hexdigest()==expected_sha:
            budget-=size
            receipts.append({'path':name,'status':'downloaded','bytes':size,'sha256':expected_sha,'expected_lfs_sha256':expected_sha})
            continue
        digest, downloaded = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024**2):
                downloaded += len(chunk)
                if downloaded > size or downloaded > budget:
                    raise ValueError('Download exceeded declared byte budget')
                digest.update(chunk)
                stream.write(chunk)
        if downloaded != size or (expected_sha and digest.hexdigest() != expected_sha):
            raise ValueError('Downloaded file failed size/SHA256 verification')
        temporary.replace(destination)
        budget -= downloaded
        receipts.append({'path': name, 'status': 'downloaded', 'bytes': downloaded,
                         'sha256': digest.hexdigest(), 'expected_lfs_sha256': expected_sha})
    return {'source': f'https://huggingface.co/datasets/{repo}', 'revision': revision,
            'license': info.get('cardData', {}).get('license'), 'files': receipts,
            'scope': f'{len(object_ids)} deterministic real-video pilot objects, not a representative benchmark; no E/nu/rho labels assumed.'}


def phystwin(root):
    repo = 'Jianghanxiao/PhysTwin'
    info = get_json(f'https://huggingface.co/api/datasets/{repo}')
    revision = info['sha']
    entries = get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}')
    entry = next(e for e in entries if e['path'] == 'data.zip')
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/data.zip'
    stream = HTTPRange(url, entry['size'])
    samples, selected = [], []
    with zipfile.ZipFile(stream) as archive:
        members = archive.infolist()
        write_json(root / 'phystwin/archive_index.json',
                   [{'name': m.filename, 'bytes': m.file_size, 'compressed': m.compress_size} for m in members])
        metadata = [m for m in members if m.filename.lower().endswith(('.json','.yaml','.yml','.txt','.pkl'))
                    and not m.is_dir() and m.file_size < 128*1024 and m.compress_size < 128*1024]
        for member in metadata[:12]:
            # Pickle is downloaded only for inventory; never deserialized.
            target = (root / 'phystwin/metadata' / member.filename).resolve()
            if not target.is_relative_to((root / 'phystwin/metadata').resolve()):
                raise ValueError('Unsafe archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(member)
            target.write_bytes(data)
            selected.append({'name': member.filename, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            if member.filename.endswith('.json'):
                value = json.loads(data)
                samples.append({'name': member.filename,
                                'top_level_keys': list(value)[:40] if isinstance(value, dict) else type(value).__name__})
    return {'source': f'https://huggingface.co/datasets/{repo}', 'revision': revision,
            'dataset_card_license': info.get('cardData', {}).get('license'),
            'archive_bytes': entry['size'], 'archive_lfs_sha256': (entry.get('lfs') or {}).get('oid'),
            'archive_members': len(members), 'metadata_candidates': len(metadata),
            'range_downloaded_bytes': stream.downloaded, 'metadata_files': selected, 'schemas': samples,
            'scope': 'ZIP central-directory and selected metadata only; CRC checked for extracted members. Full archive SHA not checked; no videos installed or material labels normalized yet.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/external_pilots')
    p.add_argument('--output', default='docs/DATA_SOURCE_PILOTS.json')
    p.add_argument('--source', choices=['all','deform360','phystwin'], default='all')
    p.add_argument('--deform360-objects',default='001-rope,004-rubber-band,008-pink-cloth,046-sponge,105-clay-cloth')
    a = p.parse_args()
    previous=json.loads(Path(a.output).read_text()) if Path(a.output).exists() else {}
    result = {'status': 'running', 'sources': previous.get('sources',{}), 'used_for_training': False}
    write_json(a.output, result)
    for name, acquire in [('deform360', lambda root:deform360(root,a.deform360_objects.split(','))), ('phystwin', phystwin)]:
        if a.source not in ('all', name):
            continue
        try:
            result['sources'][name] = {'status': 'complete', **acquire(Path(a.root).resolve())}
        except Exception as error:
            result['sources'][name] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
        write_json(a.output, result)
        print(json.dumps({name: result['sources'][name]}, ensure_ascii=False), flush=True)
    result['status'] = 'complete' if all(s['status'] == 'complete' for s in result['sources'].values()) else 'needs_repair'
    write_json(a.output, result)
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
