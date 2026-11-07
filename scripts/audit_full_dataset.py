"""Verify the official archive, extract training inputs, and audit every NPZ.

No quality-based sample filtering. All byte/CRC/schema failures stop indexing.
Video, color-field and motion statistics do not certify physical validity.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import zipfile
import numpy as np
from download_dataset import EXPECTED_SHA256, EXPECTED_SIZE, sha256


def inspect_sample(pack):
    gt = json.loads((pack.parent / "gt.json").read_text())
    run = json.loads((pack.parent / "meta/run_parameters.json").read_text())
    payload = pack.read_bytes()
    arrays, signals = {}, {}
    shape = None
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        for key in ("rgb", "flow", "stress", "force_mask", "object_mask"):
            a = z[key]  # Full decompression: inner ZIP CRC checked, including unused frames.
            if a.dtype != np.uint8 or a.ndim != 5 or a.shape[2] != 3:
                raise ValueError(f"Invalid released encoding: {pack}:{key}")
            if shape is not None and a.shape != shape:
                raise ValueError(f"Unsynchronized arrays: {pack}:{key}")
            shape = a.shape
            arrays[key] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            # Explicitly an image statistic, not optical flow in physical units.
            signals[key + "_first_last_image_mae"] = float(np.abs(a[:, -1].astype(np.int16)-a[:, 0]).mean()/255)
            if key == "object_mask":
                signals["foreground_fraction"] = float((a[:, ::8].max(2) > 127).mean())
            if key == "force_mask":
                sampled = a[:, ::8].astype(np.int16)
                gray = (sampled.max(2)-sampled.min(2)).max() == 0
                mask = sampled[:, :, 0] if gray else sampled[:, :, 0]-sampled[:, :, 1:].max(2)
                signals["contact_fraction"] = float((mask >= 128).mean())
            del a
    indices = run["render_frame_indices"]
    dt = float(run["frame_dt"])
    if len(indices) != shape[1] or dt <= 0 or any(y <= x for x, y in zip(indices, indices[1:])):
        raise ValueError(f"Invalid timing: {pack}")
    params = run["material_params"]
    substep = float(run["substep_dt"])
    if substep <= 0:
        raise ValueError(f"Invalid substep: {pack}")
    damping = float(params.get("grid_v_damping_scale", 1))
    gamma = -math.log(damping)/substep if 0 < damping < 1 else None
    labels = gt.get("regression", {})
    for key in ("E", "nu", "density"):
        if key in labels and not math.isclose(float(labels[key]), float(params[key]), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"GT/run parameter mismatch: {pack}:{key}")
    return {"sample_id": pack.parent.name, "object_id": gt["object"], "action": gt["action"],
            "material": params["material"], "arrays": arrays, "signals": signals,
            "pack_sha256": hashlib.sha256(payload).hexdigest(), "substep_dt": substep,
            "frame_dt": dt, "elapsed_seconds": (indices[-1]-indices[0])*dt,
            "grid_v_damping_scale": damping, "rpic_damping": params.get("rpic_damping"),
            "isolated_grid_damping_gamma_per_second": gamma,
            "raw_volumetric_stress_export": run.get("output_stress_volumetric", False),
            "flow_visualization_max_motion": run.get("flow_gaussian_vis_max_motion"),
            "stress_visualization_sh_blend": run.get("stress_gaussian_sh_blend")}


def run(archive, output, report, workers=16):
    archive, output, report = Path(archive), Path(output).resolve(), Path(report)
    if archive.stat().st_size != EXPECTED_SIZE or sha256(archive) != EXPECTED_SHA256:
        raise ValueError("Archive does not match the official LFS SHA256")
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        selected = [x for x in members if x.filename.endswith((".json", "/sample_pack.npz"))]
        geometry = [x.filename for x in members if x.filename.lower().endswith((".ply", ".obj", ".stl", ".glb", ".vtu"))]
    output.mkdir(parents=True, exist_ok=True)
    thread_state = threading.local()
    handles, handle_lock = [], threading.Lock()
    def extract(info):
        target = (output / info.filename).resolve()
        if not target.is_relative_to(output) or info.is_dir() or info.file_size > 512*1024**2:
            raise ValueError("Unsafe/unexpected archive member")
        if not hasattr(thread_state, "archive"):
            thread_state.archive = zipfile.ZipFile(archive)
            with handle_lock:
                handles.append(thread_state.archive)
        data = thread_state.archive.read(info)  # Always check outer CRC, even on reruns.
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".extracting")
        temporary.write_bytes(data); temporary.replace(target)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, _ in enumerate(pool.map(extract, selected), 1):
                if n % 1000 == 0:
                    print(json.dumps({"stage": "extract", "complete": n, "total": len(selected)}), flush=True)
    finally:
        for handle in handles:
            handle.close()
    packs = sorted(output.rglob("sample_pack.npz"))
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(inspect_sample, p): p for p in packs}
        for n, job in enumerate(as_completed(jobs), 1):
            try:
                records.append(job.result())
            except Exception as e:
                failures.append({"pack": str(jobs[job]), "error": repr(e)})
            if n % 100 == 0:
                print(json.dumps({"stage": "npz_audit", "complete": n, "total": len(packs), "failures": len(failures)}), flush=True)
    records.sort(key=lambda x: x["sample_id"])
    damping = defaultdict(Counter)
    for r in records:
        damping[r["material"]][str((r["grid_v_damping_scale"], r["rpic_damping"], r["substep_dt"]))] += 1
    result = {"status": "failed" if failures else "audited", "archive_sha256": EXPECTED_SHA256,
              "members": len(members), "packs": len(packs), "audited_samples": len(records),
              "gt_count": len(list(output.rglob("gt.json"))), "geometry_members": geometry,
              "objects": len({r["object_id"] for r in records}),
              "material_counts": dict(Counter(r["material"] for r in records)),
              "action_counts": dict(Counter(r["action"] for r in records)),
              "damping_by_material": {k: dict(v) for k, v in damping.items()},
              "failures": failures, "records": records,
              "scope": "Official archive SHA256; all selected outer CRCs; five full inner arrays per pack with CRC/schema checks; label/timing checks. No physical calibration or test quality scores."}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "records"}), flush=True)
    if failures:
        raise RuntimeError("Audit failures: manifest creation stopped; no silent filtering")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--archive", required=True)
    p.add_argument("--output", required=True); p.add_argument("--report", required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(); run(a.archive, a.output, a.report, a.workers)
