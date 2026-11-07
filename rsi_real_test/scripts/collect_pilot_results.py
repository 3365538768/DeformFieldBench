"""Record exploratory validation outcomes, including a train-only constant baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
from bootstrap_object_ci import compare
from fieldbridge.data import read_manifest, transform_parameters


def run(prefix,output):
    summaries={}
    for name in ("C_aux_no_feedback","D_field_bridge"):
        root=Path("runs")/(prefix+name)
        history=[]
        for path in sorted(root.glob("val_epoch_*/summary.json")):
            d=json.loads(path.read_text());history.append({"epoch":int(path.parent.name.rsplit('_',1)[-1]),"mae":d["standardized_mae"]})
        summaries[name]={"status":json.loads((root/"status.json").read_text()),
            "best_epoch":min(history,key=lambda x:x["mae"])["epoch"],"validation_curve":history,
            "best_validation":json.loads((root/"eval_val_best/summary.json").read_text()),
            "fields":json.loads((root/"field_diagnostics_val.json").read_text())}
    c=Path("runs")/(prefix+"C_aux_no_feedback")/"eval_val_best/predictions.jsonl"
    d=Path("runs")/(prefix+"D_field_bridge")/"eval_val_best/predictions.jsonl"
    rows=read_manifest("data/manifests/engineering_pilot_240.jsonl")
    stats=json.loads(Path("data/manifests/engineering_pilot_240.stats.json").read_text())
    train=[transform_parameters(r["parameters"]) for r in rows if r["split"]=="train"]
    median=np.array([np.median([y[j] for y,m in train if m[j]]) for j in range(4)])
    mean=np.array(stats["mean"]);std=np.array(stats["std"])
    baseline={}
    for name,pred in (("train_mean",mean),("train_median",median)):
        errors=[]
        for r in rows:
            if r["split"]!="val":continue
            y,m=transform_parameters(r["parameters"])
            errors.extend((np.abs(pred-y)/std)[m].tolist())
        baseline[name+"_standardized_mae"]=float(np.mean(errors))
    score_c=summaries["C_aux_no_feedback"]["best_validation"]["standardized_mae"]
    score_d=summaries["D_field_bridge"]["best_validation"]["standardized_mae"]
    result={"purpose":"engineering_pilot_only","samples":len(rows),"seed_count":1,"backbone":"DINOv2-S",
            "test_evaluated":False,"variants":summaries,"constant_baselines":baseline,
            "relative_error_reduction_D_vs_C":1-score_d/score_c,
            "exploratory_paired_object_interval":compare(c,d),
            "limitation":"Nonrandom 240-sample pilot, 14 validation clips, best epoch selected on this same validation set. Interval is descriptive only; not confirmatory evidence."}
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!="variants"},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prefix",default="engineering_pilot_240_");p.add_argument("--output",required=True)
    a=p.parse_args();run(a.prefix,a.output)
