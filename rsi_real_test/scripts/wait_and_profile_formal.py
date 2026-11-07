"""One-off bounded profile after the official giant checkpoint has arrived."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run():
    started=time.time();state=Path("docs/FORMAL_PROFILE_STATUS.json")
    def status(phase,**extra):
        d={"phase":phase,"pid":os.getpid(),"elapsed_seconds":time.time()-started,**extra}
        state.write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
    try:
        status("waiting_for_official_giant_checkpoint")
        while not Path("weights/dinov2_vitg14_reg4_pretrain.pth").exists():
            if time.time()-started>14400:raise TimeoutError("Checkpoint wait exceeded four hours")
            time.sleep(20)
        cfg=json.loads(Path("configs/p0_dinov2_giant.json").read_text())
        cfg["data"].update(manifest="data/manifests/engineering_pilot_240.jsonl",
                            stats="data/manifests/engineering_pilot_240.stats.json")
        cfg["train"].update(contact_loss="bce",purpose="pilot")
        env=os.environ.copy();env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0")
        for frames in (16,64):
            used=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
            if int(used.splitlines()[0])>1000:raise RuntimeError("GPU 0 is busy; do not displace other work")
            cfg["data"]["frames"]=frames
            path=Path(f"configs/profile_giant_{frames}frames.json")
            path.write_text(json.dumps(cfg,indent=2))
            status("profiling",frames=frames)
            subprocess.run([sys.executable,"-u","scripts/profile_formal_backbone.py","--config",str(path),
                "--output",f"docs/formal_profile_{frames}frames.json"],check=True,env=env,timeout=1800)
        status("profile_finished",note="Training samples only, no model-quality or test-set claim")
    except Exception as e:
        status("failed",error=repr(e));raise


if __name__=="__main__":
    if "--detach" in sys.argv:
        receipt=Path("docs/formal_profile_job.json")
        if receipt.exists():raise FileExistsError("Profile job already launched")
        log=Path("runs/phase_logs/formal_profile.log")
        with log.open("w") as f:
            process=subprocess.Popen([sys.executable,"-u",__file__],stdout=f,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        receipt.write_text(json.dumps({"pid":process.pid,"log":str(log),"launched_unix":time.time()},indent=2))
        print(receipt.read_text())
    else:
        run()
