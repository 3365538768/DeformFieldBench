"""Matched rerun: only contact loss changes relative to the first pilot."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run():
    receipt=Path("docs/ACTIVE_BALANCED_PILOT_JOBS.json")
    if receipt.exists():raise FileExistsError("Inspect existing pilot jobs; do not duplicate")
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
    if any(int(x)>1000 for x in gpu.splitlines()):raise RuntimeError("GPUs not idle")
    jobs=[]
    for name,devices in (("C_aux_no_feedback","0,1,2,3"),("D_field_bridge","4,5,6,7")):
        cfg=json.loads(Path("configs/pilot_"+name+".json").read_text())
        cfg["train"].update(contact_loss="balanced_bce",output="runs/engineering_pilot_240_balanced_"+name)
        path=Path("configs/pilot_balanced_"+name+".json")
        path.write_text(json.dumps(cfg,indent=2))
        env=os.environ.copy();env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1",CUDA_VISIBLE_DEVICES=devices)
        args=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=4",
              "-m","fieldbridge.train","--config",str(path)]
        log=Path("runs/phase_logs/pilot_balanced_"+name+".log")
        with log.open("w") as f:
            p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        jobs.append({"name":name,"pid":p.pid,"devices":devices,"config":str(path),"log":str(log),"launched_unix":time.time()})
        receipt.write_text(json.dumps({"jobs":jobs},indent=2))
    print(receipt.read_text())


if __name__=="__main__":run()
