"""Inspect public Kaggle ZIP central directory without fetching simulation files."""
import json
import argparse
from pathlib import Path
import urllib.request
import zipfile
from fetch_sample_subset import HTTPRange


def run(version):
    url="https://www.kaggle.com/api/v1/datasets/download/anonymous336/physical-field-material-parameter-5000?datasetVersionNumber="+str(version)
    # Kaggle's API route does not support HEAD; request a single byte and close.
    with urllib.request.urlopen(urllib.request.Request(url,headers={"Range":"bytes=0-0"}),timeout=60) as r:
        size=int(r.headers["Content-Range"].rsplit('/',1)[-1]) if r.status==206 else int(r.headers["Content-Length"])
        resolved = r.geturl()
        print(json.dumps({"archive_bytes":size,"content_type":r.headers.get("Content-Type")}),flush=True)
    stream=HTTPRange(resolved,size,cache_bust=False)
    with zipfile.ZipFile(stream) as z:
        geometry=[{"name":x.filename,"bytes":x.file_size,"compressed":x.compress_size}
                  for x in z.infolist() if x.filename.lower().endswith((".ply",".obj",".glb",".stl"))]
    d={"archive_bytes":size,"geometry_count":len(geometry),"geometry_bytes":sum(x["bytes"] for x in geometry),
       "geometry_compressed_bytes":sum(x["compressed"] for x in geometry),"geometry":geometry,
       "range_downloaded_bytes":stream.downloaded,"source":f"Kaggle version {version}, official ZIP central directory"}
    target=Path(__file__).resolve().parents[1]/f"data/metadata/kaggle_v{version}_archive_geometry.json"
    target.write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!="geometry"},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--version",type=int,default=2)
    run(p.parse_args().version)
