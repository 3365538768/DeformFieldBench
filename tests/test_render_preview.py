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


def test_contact_artifacts_require_contact_registration_and_own_root(tmp_path):
    identifier='clip_'+'c'*20;folder=tmp_path/'data/rendered/g3_contact_preflight'/identifier
    folder.mkdir(parents=True);(folder/'final.png').write_bytes(b'contact')
    docs=tmp_path/'docs';docs.mkdir()
    row=dict(id=identifier,path=str(folder.relative_to(tmp_path)),physical_tool_contact=True)
    report=docs/'G3_CONTACT_PREFLIGHT.json';report.write_text(json.dumps(dict(status='running',clips=[row])))
    catalog=preview.load_render(dict(clip='',asset=''),tmp_path)
    assert catalog['status']=='running' and catalog['clips']==[row]
    assert base64.b64decode(preview.load_render(dict(clip=identifier,asset='final.png'),tmp_path)['content_base64'])==b'contact'
    row['physical_tool_contact']=False;report.write_text(json.dumps(dict(clips=[row])))
    with pytest.raises(ValueError,match='path'):
        preview.load_render(dict(clip=identifier,asset='final.png'),tmp_path)


def test_motion_teacher_previews_require_separate_registered_root(tmp_path):
    identifier='clip_'+'d'*20;folder=tmp_path/'data/rendered/m2_motion'/identifier
    folder.mkdir(parents=True);(folder/'contact_sheet.png').write_bytes(b'motion')
    docs=tmp_path/'docs';docs.mkdir()
    row=dict(id=identifier,path=str(folder.relative_to(tmp_path)),motion_supervision=True)
    report=docs/'M2_MOTION_PREVIEW.json';report.write_text(json.dumps(dict(status='complete',clips=[row])))
    assert base64.b64decode(preview.load_render(dict(clip=identifier,asset='contact_sheet.png'),tmp_path)['content_base64'])==b'motion'
    row['motion_supervision']=False;report.write_text(json.dumps(dict(clips=[row])))
    with pytest.raises(ValueError,match='path'):preview.load_render(dict(clip=identifier,asset='contact_sheet.png'),tmp_path)
