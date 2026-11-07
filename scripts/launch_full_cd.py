"""Launch the first full-data, matched frozen DINOv2-g C/D experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, read_manifest


def run():
    receipt = Path("docs/ACTIVE_FULL_CD_JOBS.json")
    if receipt.exists():
        raise FileExistsError("Full-data C/D jobs already registered; inspect or resume them")
    audit = json.loads(Path("docs/full_dataset_audit.json").read_text())
    phase = json.loads(Path("docs/P0_PHASE_STATUS.json").read_text())
    if phase["phase"] != "p0_data_ready" or audit["failures"] or audit["audited_samples"] != 4914:
        raise RuntimeError("Full-data audit/indexing not complete")
    rows = read_manifest("data/manifests/deformfield_v1.jsonl")
    published = json.loads(Path("data/metadata/train_test_split.cleaned.json").read_text())
    if {Path(r["pack"]).parent.name for r in rows if r["split"] == "test"} != set(published["test_ids"]):
        raise ValueError("Published test IDs changed")
    split_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    base = json.loads(Path("configs/p0_dinov2_giant.json").read_text())
    prepared = []
    for name, feedback, devices in (("C_aux_no_feedback", "none", "0,1,2,3"),
                                     ("D_field_bridge", "fields", "4,5,6,7")):
        cfg = json.loads(json.dumps(base))
        cfg["model"]["feedback"] = feedback
        cfg["train"].update(seed=17, workers=3, contact_loss="bce", purpose="research",
                             output="runs/full_giant16_" + name + "_seed17")
        path = Path("configs/full_giant16_" + name + "_seed17.json")
        if path.exists() or Path(cfg["train"]["output"]).exists():
            raise FileExistsError("Use existing full-data configs/output instead of overwriting")
        path.write_text(json.dumps(cfg, indent=2))
        parsed = load_config(path)
        for split in ("train", "val"):
            ds = MaterialDataset(parsed.data, split, parsed.model.field_size, parsed.model.field_mode)
            if not len(ds):
                raise ValueError("Empty development split")
        prepared.append((name, devices, path))
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if len(gpu.splitlines()) != 8 or any(int(x.strip()) > 1000 for x in gpu.splitlines()):
        raise RuntimeError("Eight idle GPUs required; do not displace other work")
    logs = Path("runs/phase_logs"); logs.mkdir(parents=True, exist_ok=True)
    record = {"purpose": "first full-data controlled experiment; one seed, no test evaluation",
              "split_counts": split_counts, "audit_failures": 0,
              "manifest_sha256": hashlib.sha256(Path(base["data"]["manifest"]).read_bytes()).hexdigest(),
              "epochs": 80, "field_only_epochs": 10, "views": 3, "frames": 16,
              "effective_batch_size_per_variant": 8,
              "note": "16-frame development protocol, not exact 64-frame paper reproduction. Plain BCE retained to isolate scale/data from loss changes.",
              "jobs": []}
    for name, devices, config in prepared:
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=devices)
        args = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                "-m", "fieldbridge.train", "--config", str(config)]
        log = logs / ("full_giant16_" + name + "_seed17.log")
        with log.open("w") as f:
            process = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        record["jobs"].append({"name": name, "pid": process.pid, "devices": devices,
                               "config": str(config), "log": str(log), "launched_unix": time.time()})
        receipt.write_text(json.dumps(record, indent=2))
    status_path = Path("docs/BUILD_STATUS.json")
    status = json.loads(status_path.read_text())
    status["data"].update(full_dataset_downloaded=True, full_schema_crc_audit="passed",
                          original_test_preserved=True, split_counts=split_counts)
    status["models"]["dinov2_giant"] = "official checkpoint downloaded, CRC checked, strict model loading and 16/64-frame profiling passed"
    status["long_running_training_active"] = True
    status["active_jobs"] = ["full-data C_aux_no_feedback seed17, GPUs 0-3", "full-data D_field_bridge seed17, GPUs 4-7"]
    status["p0_phase"] = "first_full_data_cd_running"
    status["note"] = "Full CRC/schema/timing audit passed. First full-data controlled training started; no test-set evaluation or superiority evidence yet."
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in {"records", "damping_by_material"}}, indent=2))
    print(receipt.read_text())


if __name__ == "__main__":
    run()
