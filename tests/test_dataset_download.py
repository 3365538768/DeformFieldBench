"""Real local HTTP range responses exercise retries and final integrity checks."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import threading

import pytest

spec=importlib.util.spec_from_file_location('dataset_download',Path(__file__).parents[1]/'scripts/download_dataset.py')
download=importlib.util.module_from_spec(spec);spec.loader.exec_module(download)
pytestmark=pytest.mark.skipif(not hasattr(os,'pwrite'),reason='Parallel downloader runs on Linux')


@pytest.fixture
def endpoint(monkeypatch):
    payload=bytes(range(256))*4096
    state={'requests':[],'fail_once':False,'wrong_range':False}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state['requests'].append(self.path)
            if state['fail_once']:
                state['fail_once']=False;self.send_error(403);return
            first,last=map(int,self.headers['Range'].removeprefix('bytes=').split('-'))
            self.send_response(206)
            self.send_header('Content-Range',f'bytes {first}-{last}/{len(payload)+(1 if state["wrong_range"] else 0)}')
            self.send_header('Content-Length',str(last-first+1));self.end_headers()
            try:self.wfile.write(payload[first:last+1])
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    monkeypatch.setattr(download,'URL',f'http://127.0.0.1:{server.server_port}/archive')
    monkeypatch.setattr(download,'EXPECTED_SIZE',len(payload))
    monkeypatch.setattr(download,'EXPECTED_SHA256',hashlib.sha256(payload).hexdigest())
    try:yield payload,state
    finally:server.shutdown();server.server_close()


def test_expired_redirect_retry_uses_new_nonce_and_checks_full_hash(tmp_path,endpoint):
    payload,state=endpoint;state['fail_once']=True
    target=tmp_path/'dataset.zip'
    download.download_parallel(target,workers=2)
    assert target.read_bytes()==payload
    assert len(state['requests'])==2 and state['requests'][0]!=state['requests'][1]
    assert json.loads(target.with_suffix('.download.json').read_text())['status']=='verified'


def test_invalid_content_range_never_becomes_a_final_archive(tmp_path,endpoint):
    _,state=endpoint;state['wrong_range']=True
    target=tmp_path/'dataset.zip'
    with pytest.raises(RuntimeError,match='ValueError'):
        download.download_parallel(target,workers=2)
    assert not target.exists()
    assert len(state['requests'])==4


def test_resumed_chunk_receipt_cannot_bypass_archive_integrity(tmp_path,endpoint):
    payload,_=endpoint;target=tmp_path/'dataset.zip'
    chunks=tmp_path/'dataset.zip.chunks';chunks.mkdir()
    (chunks/'00000.json').write_text(json.dumps({'bytes':len(payload),'sha256':'wrong'}))
    with pytest.raises(ValueError,match='SHA256'):
        download.download_parallel(target,workers=2)
    assert not target.exists()
