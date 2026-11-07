"""Validate Gaussian PLY arrays and object coverage; no assumed metre scale."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def run(root, split, output):
    types={"float":"<f4","double":"<f8","uchar":"u1","uint8":"u1","int":"<i4","uint":"<u4"}
    records=[]
    for path in sorted(Path(root).rglob("*.ply")):
        payload=path.read_bytes();marker=b"end_header\n";end=payload.find(marker)
        if end<0:raise ValueError(f"Missing PLY header: {path}")
        offset=end+len(marker);header=payload[:offset].decode("ascii")
        if "format binary_little_endian 1.0" not in header:raise ValueError("Unexpected PLY encoding")
        lines=header.splitlines()
        if sum(l.startswith("element ") for l in lines)!=1:raise ValueError("Only Gaussian vertex PLY expected")
        n=int(next(l.rsplit(" ",1)[-1] for l in lines if l.startswith("element vertex ")))
        columns=[]
        for line in lines:
            if line.startswith("property "):
                _,kind,name=line.split();columns.append((name,types[kind]))
        dtype=np.dtype(columns)
        if n<=0 or len(payload)-offset!=n*dtype.itemsize:raise ValueError(f"PLY byte length mismatch: {path}")
        a=np.frombuffer(payload,dtype=dtype,offset=offset,count=n)
        for name in a.dtype.names:
            if not np.isfinite(a[name]).all():raise ValueError(f"Nonfinite {name}: {path}")
        xyz=np.stack([a[k] for k in ("x","y","z")],1)
        required={"opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"}
        if not required.issubset(a.dtype.names):raise ValueError(f"Missing Gaussian attributes: {path}")
        scales=np.stack([a[f"scale_{i}"] for i in range(3)],1)
        rotations=np.stack([a[f"rot_{i}"] for i in range(4)],1)
        records.append({"object_id":path.stem,"points":n,"sha256":hashlib.sha256(payload).hexdigest(),
            "coordinate_bbox_min":xyz.min(0).tolist(),"coordinate_bbox_max":xyz.max(0).tolist(),
            "coordinate_extent":np.ptp(xyz,axis=0).tolist(),
            "stored_log_scale_min_max":[float(scales.min()),float(scales.max())],
            "stored_opacity_min_max":[float(a["opacity"].min()),float(a["opacity"].max())],
            "quaternion_norm_min_max":[float(np.linalg.norm(rotations,axis=1).min()),float(np.linalg.norm(rotations,axis=1).max())],
            "length_scale_m":None,"solid_volume_validated":False})
    expected=set(json.loads(Path(split).read_text())["stats"]["object_counts_all"])
    found={r["object_id"] for r in records}
    if len(found)!=len(records):raise ValueError("Duplicate geometry object IDs")
    result={"status":"geometry_arrays_validated","assets":len(records),"dataset_objects":len(expected),
        "covered_dataset_objects":len(expected&found),"missing_objects":sorted(expected-found),
        "extra_objects":sorted(found-expected),"records":records,
        "limits":"Finite binary Gaussian arrays, coverage and hashes only; not calibrated physical length, volume, mass or constitutive response."}
    Path(output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
    if expected-found:raise ValueError("Dataset object geometries still missing")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("root","split","output"):p.add_argument("--"+name,required=True)
    a=p.parse_args();run(a.root,a.split,a.output)
