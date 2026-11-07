"""Validation-only foreground/contact diagnostics, without GT inference inputs."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate, model_inputs
from fieldbridge.evaluation import move_batch
from fieldbridge.model import MaterialFieldModel


@torch.no_grad()
def run(config, checkpoint, output):
    cfg = load_config(config)
    ds = MaterialDataset(cfg.data, "val", cfg.model.field_size, cfg.model.field_mode)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt["stats"] != ds.codec.stats or ckpt["config"]["model"] != cfg.dump()["model"]:
        raise ValueError("Checkpoint/validation contract mismatch")
    model = MaterialFieldModel(cfg.model).cuda().eval()
    model.load_state_dict(ckpt["model"], strict=True)
    tp=fp=fn=positive=negative=0
    flow_num=flow_den=stress_num=stress_den=0.
    positive_bce=negative_bce=0.
    fc=3 if cfg.model.field_mode == "legacy_visual" else 2
    for batch in DataLoader(ds,batch_size=1,collate_fn=collate):
        b=move_batch(batch,"cuda")
        p=model(**model_inputs(b))
        valid=b["field_mask"][:,:,:,:1]>0
        gt=b["fields"][:,:,:,:1]>.5
        pred=p["fields"][:,:,:,:1]>.5
        tp+=int((pred & gt & valid).sum())
        fp+=int((pred & ~gt & valid).sum())
        fn+=int((~pred & gt & valid).sum())
        positive+=int((gt&valid).sum());negative+=int((~gt&valid).sum())
        bce=torch.nn.functional.binary_cross_entropy_with_logits(p["contact_logits"].float(),gt.float(),reduction="none")
        positive_bce+=float(bce[gt&valid].sum());negative_bce+=float(bce[~gt&valid].sum())
        err=(p["fields"]-b["fields"]).abs()*b["field_mask"]
        flow_num+=float(err[:,:,:,1:1+fc].sum());flow_den+=float(b["field_mask"][:,:,:,1:1+fc].sum())
        stress_num+=float(err[:,:,:,1+fc:].sum());stress_den+=float(b["field_mask"][:,:,:,1+fc:].sum())
    ratio=lambda x,y: x/y if y else None
    result={"split":"val","samples":len(ds),"contact_gt_positive_fraction":ratio(positive,positive+negative),
        "contact_iou":ratio(tp,tp+fp+fn),"contact_precision":ratio(tp,tp+fp),"contact_recall":ratio(tp,tp+fn),
        "contact_positive_bce":ratio(positive_bce,positive),"contact_negative_bce":ratio(negative_bce,negative),
        "foreground_flow_encoded_mae":ratio(flow_num,flow_den),"foreground_stress_encoded_mae":ratio(stress_num,stress_den),
        "note":"Contact support and visual-field errors only; no SI force/stress/velocity claim."}
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("config","checkpoint","output"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args();run(a.config,a.checkpoint,a.output)
