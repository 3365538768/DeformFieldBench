"""One-off P0 job: wait for the active download, audit, and index. No test evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(timeout):
    state = Path("docs/P0_PHASE_STATUS.json")
    started = time.time()
    def status(phase, **extra):
        d = {"phase": phase, "elapsed_seconds": time.time()-started, "pid": os.getpid(), **extra}
        state.write_text(json.dumps(d, indent=2));print(json.dumps(d), flush=True)
    def call(args):
        subprocess.run([sys.executable, *args], check=True)
    try:
        status("waiting_for_verified_archive")
        while not Path("data/archives/dataset.zip").exists():
            if time.time()-started > timeout:
                raise TimeoutError("Download wait exceeded explicit job timeout")
            time.sleep(20)
        status("full_crc_schema_timing_audit")
        call(["scripts/audit_full_dataset.py", "--archive", "data/archives/dataset.zip", "--output", "data/deformfield",
              "--report", "docs/full_dataset_audit.json", "--workers", "16"])
        status("creating_original_test_preserving_manifest")
        manifest = "data/manifests/deformfield_v1.jsonl"
        call(["scripts/import_deformfield.py", "--root", "data/deformfield", "--output", manifest,
              "--original-split", "data/metadata/train_test_split.cleaned.json"])
        call(["-m", "fieldbridge.cli", "audit", manifest])
        call(["-m", "fieldbridge.cli", "fit-stats", manifest, "data/manifests/deformfield_v1.stats.json"])
        build_path = Path("docs/BUILD_STATUS.json")
        build = json.loads(build_path.read_text())
        build["data"].update(full_dataset_downloaded=True, full_schema_crc_audit="passed",
            manifest=manifest, original_test_preserved=True)
        build["p0_phase"] = "data_ready"
        build_path.write_text(json.dumps(build, indent=2))
        status("p0_data_ready", note="No model test-set evaluation has been performed. Formal training is a separate phase.")
    except Exception as e:
        status("failed", error=repr(e))
        raise


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--timeout-seconds",type=int,default=14400)
    run(p.parse_args().timeout_seconds)
