"""Artifact preview must not expose arbitrary remote files, including symlinks."""
import base64
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('render_preview',Path(__file__).parents[1]/'monitor/render_remote.py')
preview=importlib.util.module_from_spec(spec);spec.loader.exec_module(preview)


def test_registered_render_images_only(tmp_path):
    identifier='clip_'+'a'*20;folder=tmp_path/'data/rendered/g3_pilot'/identifier
    folder.mkdir(parents=True);(folder/'final.png').write_bytes(b'PNG-placeholder')
    docs=tmp_path/'docs';docs.mkdir()
    row=dict(id=identifier,path=str(folder.relative_to(tmp_path)))
    (docs/'G3_RENDER_QUALITY.json').write_text(json.dumps(dict(status='complete',clips=[row])))
    assert preview.load_render(dict(clip='',asset=''),tmp_path)['clips']==[row]
    result=preview.load_render(dict(clip=identifier,asset='final.png'),tmp_path)
    assert base64.b64decode(result['content_base64'])==b'PNG-placeholder'
    for clip,asset in [(identifier,'../../etc/passwd'),('clip_'+'b'*20,'final.png'),('../../etc/passwd','final.png')]:
        with pytest.raises(ValueError):preview.load_render(dict(clip=clip,asset=asset),tmp_path)
    row['path']='outside'
    (docs/'G3_RENDER_QUALITY.json').write_text(json.dumps(dict(clips=[row])))
    with pytest.raises(ValueError,match='path'):preview.load_render(dict(clip=identifier,asset='final.png'),tmp_path)
