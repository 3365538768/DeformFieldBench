"""Extract only geometry from official Kaggle version 2 by byte ranges and CRC."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import urllib.request
import zipfile
from fetch_sample_subset import HTTPRange


def run():
    root=Path(__file__).resolve().parents[1]
    output=(root/"data/geometry/kaggle_v2").resolve();output.mkdir(parents=True,exist_ok=True)
    url="https://www.kaggle.com/api/v1/datasets/download/anonymous336/physical-field-material-parameter-5000?datasetVersionNumber=2"
    with urllib.request.urlopen(urllib.request.Request(url,headers={"Range":"bytes=0-0"}),timeout=60) as r:
        size=int(r.headers["Content-Range"].rsplit('/',1)[-1]);resolved=r.geturl()
    with zipfile.ZipFile(HTTPRange(resolved,size,cache_bust=False)) as z:
        assets=[x for x in z.infolist() if x.filename.lower().endswith(".ply")]
    state=threading.local();handles=[];lock=threading.Lock()
    def fetch(info):
        if not hasattr(state,"archive"):
            state.archive=zipfile.ZipFile(HTTPRange(resolved,size,cache_bust=False))
            with lock:handles.append(state.archive)
        payload=state.archive.read(info)
        target=(output/info.filename).resolve()
        if not target.is_relative_to(output):raise ValueError("Unsafe ZIP path")
        header_end=payload.find(b"end_header\n")
        if header_end < 0 or header_end>100000:raise ValueError("Missing PLY header")
        header=payload[:header_end+11].decode("ascii")
        if not header.startswith("ply\n") or "element vertex" not in header:
            raise ValueError("Unexpected PLY schema")
        vertices=int(next(l.split()[-1] for l in header.splitlines() if l.startswith("element vertex ")))
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix(".extracting");temp.write_bytes(payload);temp.replace(target)
        return {"object_id":target.stem,"archive_member":info.filename,"local_relative":str(target.relative_to(root)),
            "bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),"zip_crc32":f"{info.CRC:08x}",
            "vertices":vertices,"gaussian_fields_present":all(x in header for x in ("opacity","scale_0","rot_0","f_dc_0"))}
    records=[]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for job in as_completed([pool.submit(fetch,x) for x in assets]):
                records.append(job.result())
                if len(records)%10==0:print(json.dumps({"geometry_downloaded":len(records),"total":len(assets)}),flush=True)
    finally:
        for handle in handles:handle.close()
    original=json.loads((root/"data/metadata/train_test_split.cleaned.json").read_text())
    expected=set(original["stats"]["object_counts_all"])
    found={r["object_id"] for r in records}
    receipt={"source_url":"https://www.kaggle.com/datasets/anonymous336/physical-field-material-parameter-5000/versions/2",
        "archive_size":size,"geometry_count":len(records),"geometry_bytes":sum(r["bytes"] for r in records),
        "expected_dataset_objects":len(expected),"covered_objects":len(expected&found),
        "missing_objects":sorted(expected-found),"extra_objects":sorted(found-expected),
        "records":sorted(records,key=lambda r:r["object_id"]),"zip_crc_checked":True,
        "note":"Gaussian surface/appearance assets; not solid volume/scale calibration. Matching IDs alone do not prove identical generation geometry."}
    path=root/"data/metadata/kaggle_geometry_receipt.json"
    path.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!="records"},indent=2))


if __name__=="__main__":run()
