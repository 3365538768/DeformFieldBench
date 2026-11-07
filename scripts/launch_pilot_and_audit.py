"""Launch this phase's bounded, one-off jobs and record their PIDs/logs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run():
    jobs_path = Path("docs/ACTIVE_PHASE_JOBS.json")
    if jobs_path.exists():
        raise FileExistsError("Phase launch receipt exists; inspect the existing jobs instead of launching duplicates")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if len(gpu.splitlines()) != 8 or any(int(x.strip()) > 1000 for x in gpu.splitlines()):
        raise RuntimeError("The eight GPUs are no longer idle; do not displace existing work")
    logs = Path("runs/phase_logs");logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    def launch(name, args, devices=""):
        env = os.environ.copy();env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices
        log = logs / (name + ".log")
        with log.open("w") as f:
            process = subprocess.Popen([sys.executable, *args], stdout=f, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        jobs.append({"name": name, "pid": process.pid, "log": str(log), "gpu_devices": devices,
                     "args": args, "launched_unix": time.time()})
        jobs_path.write_text(json.dumps({"jobs": jobs}, indent=2))
    launch("p0_full_audit", ["-u", "scripts/run_p0_pipeline.py", "--timeout-seconds", "14400"])
    for name, devices in (("C_aux_no_feedback", "0,1,2,3"), ("D_field_bridge", "4,5,6,7")):
        launch("pilot_"+name, ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "-m", "fieldbridge.train", "--config", "configs/pilot_"+name+".json"], devices)
    print(jobs_path.read_text())


if __name__ == "__main__":
    run()
