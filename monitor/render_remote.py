"""Read only registered, completed render artifacts; never accept user paths."""
import base64
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(os.environ.get('FIELDBRIDGE_ROOT','/mnt/afs_engine/linrui/rsi_real_test'))


def load_render(request,root=ROOT):
    if set(request)!={'clip','asset'} or not all(isinstance(v,str) for v in request.values()):
        raise ValueError('Expected clip ID and artifact name')
    report=root/'docs/G3_RENDER_QUALITY.json'
    data=json.loads(report.read_text()) if report.exists() else {'status':'preparing','clips':[]}
    identifier,asset=request['clip'],request['asset']
    if not identifier and not asset:
        return {k:v for k,v in data.items() if k!='quasistatic_identifiability'}
    if not re.fullmatch(r'clip_[0-9a-f]{20}',identifier) or asset not in {'preview.gif','contact_sheet.png','final.png'}:
        raise ValueError('Invalid registered render artifact')
    row=next((r for r in data.get('clips',[]) if r['id']==identifier),None)
    if row is None:raise ValueError('Clip not registered')
    folder=(root/row['path']).resolve();allowed=(root/'data/rendered/g3_pilot').resolve()
    path=(folder/asset).resolve()
    if not folder.is_relative_to(allowed) or not path.is_relative_to(folder):raise ValueError('Invalid artifact path')
    if path.stat().st_size>8*1024*1024:raise ValueError('Artifact exceeds preview limit')
    return dict(status='ready',mime='image/gif' if asset.endswith('.gif') else 'image/png',
                content_base64=base64.b64encode(path.read_bytes()).decode('ascii'))


if __name__=='__main__':
    try:print(json.dumps(load_render(json.loads(sys.stdin.read())),allow_nan=False,separators=(',',':')))
    except Exception as exc:
        print(json.dumps({'status':'error','error':type(exc).__name__}));raise SystemExit(1)
