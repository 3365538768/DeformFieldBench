"""Resumable official archive download with SHA256 verification.

The proxy is an explicit CLI option; credentials are neither read nor logged.
The original archive is immutable; extraction is a separate checked step.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

URL = "https://huggingface.co/datasets/HandsomeHusky/DeformFieldBench/resolve/main/dataset.zip"
EXPECTED_SIZE = 41425509436
EXPECTED_SHA256 = "746934865c0b45cd5cf48232f21ab70fd2b5590ea8c0ca225ebfd48d57492050"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def download(output, proxy=""):
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    receipt = target.with_suffix(".download.json")
    if target.exists():
        if target.stat().st_size != EXPECTED_SIZE or sha256(target) != EXPECTED_SHA256:
            raise ValueError("Existing archive fails integrity check")
        print("Existing verified archive", flush=True)
        return
    handler = urllib.request.ProxyHandler({"https": proxy, "http": proxy}) if proxy else urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(handler)
    started = time.time()
    failures = 0
    while (partial.stat().st_size if partial.exists() else 0) < EXPECTED_SIZE:
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "FieldBridge-dataset-audit/0.1"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(URL + f"?download=true&fieldbridge_full_offset={start}", headers=headers)
        try:
            with opener.open(request, timeout=90) as r:
                if start:
                    expected = f"bytes {start}-{EXPECTED_SIZE-1}/{EXPECTED_SIZE}"
                    if r.status != 206 or r.headers.get("Content-Range") != expected:
                        raise ValueError("Resume response range mismatch")
                elif r.status != 200:
                    raise ValueError("Initial archive request must return the full object")
                with partial.open("ab") as f:
                    checkpoint = start
                    while True:
                        block = r.read(4 * 1024**2)
                        if not block:
                            break
                        f.write(block)
                        current = f.tell()
                        if current > EXPECTED_SIZE:
                            raise ValueError("Unexpected oversized archive")
                        if current - checkpoint >= 128 * 1024**2:
                            f.flush()
                            status = {"status": "downloading", "bytes": current, "total": EXPECTED_SIZE,
                                      "elapsed_seconds": round(time.time()-started, 2)}
                            receipt.write_text(json.dumps(status, indent=2))
                            print(json.dumps(status), flush=True)
                            checkpoint = current
            failures = 0
        except (OSError, TimeoutError) as error:
            failures += 1
            print(json.dumps({"retry": failures, "error": str(error)}), flush=True)
            if failures >= 5:
                raise
    receipt.write_text(json.dumps({"status": "verifying_sha256", "bytes": EXPECTED_SIZE}))
    digest = sha256(partial)
    if digest != EXPECTED_SHA256:
        raise ValueError("Full archive SHA256 mismatch; partial preserved for diagnosis")
    partial.replace(target)
    status = {"status": "verified", "url": URL, "bytes": EXPECTED_SIZE, "sha256": digest,
              "elapsed_seconds": time.time()-started}
    receipt.write_text(json.dumps(status, indent=2)); print(json.dumps(status), flush=True)


def download_parallel(output, proxy="", workers=8):
    target = Path(output).resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".segmented.partial")
    chunks = target.parent / (target.name + ".chunks"); chunks.mkdir(exist_ok=True)
    receipt = target.with_suffix(".download.json")
    chunk_size = 64 * 1024**2
    count = (EXPECTED_SIZE + chunk_size - 1) // chunk_size
    if target.exists():
        if target.stat().st_size != EXPECTED_SIZE or sha256(target) != EXPECTED_SHA256:
            raise ValueError("Existing final archive is invalid")
        return
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, EXPECTED_SIZE)
    started = time.time()
    # Recover complete ranges already downloaded by the sequential helper.
    prefix = target.with_suffix(target.suffix + ".partial")
    if prefix.exists():
        with prefix.open("rb") as f:
            for i in range(prefix.stat().st_size // chunk_size):
                block = f.read(chunk_size)
                done = chunks / f"{i:05d}.json"
                if not done.exists():
                    written = os.pwrite(fd, block, i * chunk_size)
                    if written != len(block):raise IOError("Short prefix write")
                    done.write_text(json.dumps({"bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()}))
    def fetch(i):
        start, end = i*chunk_size, min((i+1)*chunk_size, EXPECTED_SIZE)-1
        done = chunks / f"{i:05d}.json"
        if done.exists():return i, end-start+1
        error = None
        for attempt in range(4):
            try:
                handler = urllib.request.ProxyHandler({"https":proxy,"http":proxy}) if proxy else urllib.request.ProxyHandler({})
                opener = urllib.request.build_opener(handler)
                url = URL + f"?download=true&fieldbridge_chunk={start}-{end}"
                request = urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}","User-Agent":"FieldBridge/0.1"})
                offset, digest = start, hashlib.sha256()
                with opener.open(request,timeout=90) as r:
                    if r.status!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{EXPECTED_SIZE}":
                        raise ValueError("Chunk range mismatch")
                    while offset<=end:
                        block=r.read(min(4*1024**2,end-offset+1))
                        if not block:raise IOError("Truncated chunk")
                        if os.pwrite(fd,block,offset)!=len(block):raise IOError("Short pwrite")
                        offset+=len(block);digest.update(block)
                done.write_text(json.dumps({"bytes":end-start+1,"sha256":digest.hexdigest()}))
                return i,end-start+1
            except Exception as e:
                error=e
        raise RuntimeError(f"Chunk {i} failed: {error}")
    try:
        completed_bytes=0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(fetch,i) for i in range(count)]
            for future in as_completed(futures):
                _, size=future.result();completed_bytes+=size
                status={"status":"downloading_chunks","bytes":completed_bytes,"total":EXPECTED_SIZE,
                        "complete_chunks":len(list(chunks.glob('*.json'))),"total_chunks":count,
                        "workers":workers,"elapsed_seconds":round(time.time()-started,2)}
                receipt.write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    receipt.write_text(json.dumps({"status":"verifying_sha256","bytes":EXPECTED_SIZE}))
    digest=sha256(partial)
    if digest!=EXPECTED_SHA256:raise ValueError("Full archive SHA256 mismatch")
    partial.replace(target)
    status={"status":"verified","url":URL,"bytes":EXPECTED_SIZE,"sha256":digest,"elapsed_seconds":time.time()-started}
    receipt.write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--proxy",default="")
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    if a.workers>1:download_parallel(a.output,a.proxy,a.workers)
    else:download(a.output,a.proxy)
