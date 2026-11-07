"""Validation intervention on bridge inputs, keeping checkpoint and RGB fixed.

Zero/reversed inputs may be out of distribution. Sensitivity is a dependency
diagnostic, not an accuracy improvement or a causal physical identification claim.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from fieldbridge.config import load_config, ModelConfig
from fieldbridge.data import MaterialDataset, ShuffledOracleDataset, collate
from fieldbridge.evaluation import move_batch
from fieldbridge.local_model import build_model
from fieldbridge.runtime import atomic_json, forward_batch, load_weights, sha256


@torch.no_grad()
def run(config, checkpoint, output, fp32=False):
    cfg = load_config(config)
    if fp32:cfg.train.val_bf16=False
    ds = MaterialDataset(cfg.data, 'val', cfg.model.field_size, cfg.model.field_mode)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if asdict(ModelConfig(**ckpt['config']['model'])) != cfg.dump()['model'] or ckpt['stats'] != ds.codec.stats:
        raise ValueError('Checkpoint/model/statistics mismatch')
    if ckpt['manifest_sha256'] != sha256(cfg.data.manifest):
        raise ValueError('Checkpoint/manifest mismatch')
    backbone = sha256(cfg.model.weights) if cfg.model.weights else None
    if ds.prepared and (ds.prepared['backbone_checkpoint_sha256'] != backbone or
                        ds.prepared['encoder_source_sha256'] != sha256(Path(__file__).parents[1]/'src/fieldbridge/model.py')):
        raise ValueError('Cached encoder provenance mismatch')
    if cfg.model.feedback == 'oracle_shuffled':ds = ShuffledOracleDataset(ds)
    device = torch.device(cfg.train.device); model = build_model(cfg.model).to(device).eval()
    load_weights(model, ckpt, backbone)
    names = ['original', 'zero_bridge', 'reverse_time_bridge', 'batch_roll_bridge']
    metrics = {name:dict(error=0.,change=0.,max_change=0.,count=0.) for name in names}
    ids=[]
    for batch in DataLoader(ds,batch_size=cfg.train.val_batch_size,collate_fn=collate,num_workers=cfg.train.val_workers):
        b=move_batch(batch,device);ids.extend(b['sample_id'])
        shape=tuple(b['valid'].shape)
        reference=None
        for name in names:
            def intervention(module, inputs):
                x=inputs[0]
                if name=='zero_bridge':return (torch.zeros_like(x),)
                if name=='batch_roll_bridge':return (x.reshape(*shape,*x.shape[1:]).roll(1,0).reshape_as(x),)
                return (x.reshape(*shape,*x.shape[1:]).flip(2).reshape_as(x),)
            handle=model.field_encoder[0].register_forward_pre_hook(intervention) if name!='original' else None
            try:
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=cfg.train.val_bf16 and device.type=='cuda'):
                    prediction=forward_batch(model,b,cfg.model.feedback)['mean'].float()
            finally:
                if handle:handle.remove()
            if reference is None:reference=prediction
            mask=b['target_mask'];change=(prediction-reference).abs()
            value=metrics[name]
            value['error']+=float(((prediction-b['target']).abs()*mask).sum())
            value['change']+=float((change*mask).sum());value['max_change']=max(value['max_change'],float(change.max()))
            value['count']+=float(mask.sum())
    if len(ids)!=len(ds) or len(set(ids))!=len(ds):raise ValueError('Validation coverage mismatch')
    for value in metrics.values():
        value['standardized_mae']=value.pop('error')/max(value['count'],1)
        value['mean_prediction_change_z']=value.pop('change')/max(value['count'],1)
        value['max_prediction_change_z']=value.pop('max_change')
    result={'samples':len(ids),'split':'val','checkpoint':checkpoint,'checkpoint_epoch':ckpt['epoch']+1,
            'precision':'fp32' if not cfg.train.val_bf16 else 'bf16',
            'feedback':cfg.model.feedback,'metrics':metrics,
            'note':'Same RGB/checkpoint, intervention only before field encoder. OOD dependency diagnostic; not a physical causal claim or independent test result.'}
    atomic_json(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['config','checkpoint','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--fp32',action='store_true')
    args=parser.parse_args();run(args.config,args.checkpoint,args.output,args.fp32)
