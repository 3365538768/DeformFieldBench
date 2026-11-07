"""Atomic resumable download from an explicit official DINOv2 URL.

The final SHA256 is recorded as provenance, not claimed as an upstream checksum.
Structural/key validation happens when FrameEncoder loads the official model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def run(variant, proxy):
    url=f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{variant}/dinov2_{variant}_reg4_pretrain.pth"
    target=Path(f"weights/dinov2_{variant}_reg4_pretrain.pth")
    target.parent.mkdir(exist_ok=True)
    part=target.with_suffix(".part")
    receipt=target.with_suffix(".download.json")
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({"https":proxy,"http":proxy}))
    with opener.open(urllib.request.Request(url,method="HEAD"),timeout=90) as response:
        total=int(response.headers["Content-Length"])
    if target.exists():
        if target.stat().st_size != total:
            raise ValueError("Existing checkpoint has unexpected size")
        print("Existing checkpoint; model loader will validate structure",flush=True)
        return
    started=time.time()
    def status(phase, size):
        record={"status":phase,"url":url,"bytes":size,"total":total,"elapsed_seconds":time.time()-started}
        receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    for attempt in range(8):
        current=part.stat().st_size if part.exists() else 0
        if current==total:
            break
        if current>total:
            raise ValueError("Oversized partial checkpoint")
        headers={"User-Agent":"FieldBridge/0.1"}
        if current:
            headers["Range"]=f"bytes={current}-"
        try:
            req=urllib.request.Request(url+f"?fieldbridge_offset={current}",headers=headers)
            with opener.open(req,timeout=90) as response:
                if current and (response.status!=206 or response.headers.get("Content-Range")!=f"bytes {current}-{total-1}/{total}"):
                    raise ValueError("Checkpoint resume range mismatch")
                if not current and response.status!=200:
                    raise ValueError("Full checkpoint response must be 200")
                with part.open("ab") as f:
                    checkpoint=current
                    while True:
                        data=response.read(4*1024**2)
                        if not data:
                            break
                        f.write(data);current+=len(data)
                        if current>total:
                            raise ValueError("Oversized checkpoint response")
                        if current-checkpoint>=128*1024**2:
                            f.flush();status("downloading",current);checkpoint=current
        except (OSError,TimeoutError) as e:
            print(json.dumps({"attempt":attempt+1,"error":repr(e)}),flush=True)
    if not part.exists() or part.stat().st_size != total:
        raise RuntimeError("Checkpoint remains incomplete; preserved partial")
    digest=hashlib.sha256()
    with part.open("rb") as f:
        for block in iter(lambda:f.read(16*1024**2),b""):
            digest.update(block)
    # Check the PyTorch archive CRC before publishing the path used by configs.
    import zipfile
    if not zipfile.is_zipfile(part):
        raise ValueError("Expected the official PyTorch zip checkpoint")
    with zipfile.ZipFile(part) as z:
        bad=z.testzip()
    if bad:
        raise ValueError(f"Checkpoint CRC failure: {bad}")
    part.replace(target)
    receipt.write_text(json.dumps({"status":"downloaded_crc_checked","url":url,"bytes":total,
        "sha256":digest.hexdigest(),"upstream_sha256_known":False,
        "elapsed_seconds":time.time()-started},indent=2))
    print(receipt.read_text(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--variant",choices=["vits14","vitb14","vitl14","vitg14"],default="vitg14")
    p.add_argument("--proxy",required=True)
    a=p.parse_args();run(a.variant,a.proxy)
