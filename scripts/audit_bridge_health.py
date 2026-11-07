"""Read-only validation audit of bridge gate collapse and content activations."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from fieldbridge.config import load_config, ModelConfig
from fieldbridge.data import MaterialDataset, ShuffledOracleDataset, collate
from fieldbridge.evaluation import move_batch
from fieldbridge.local_model import build_model
from fieldbridge.runtime import atomic_json, forward_batch, load_weights, sha256


@torch.no_grad()
def run(config, checkpoint, output):
    cfg=load_config(config);torch.set_num_threads(1)
    ds=MaterialDataset(cfg.data,'val',cfg.model.field_size,cfg.model.field_mode)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if asdict(ModelConfig(**ck['config']['model']))!=cfg.dump()['model'] or ck['stats']!=ds.codec.stats:
        raise ValueError('Model/statistics mismatch')
    backbone=sha256(cfg.model.weights) if cfg.model.weights else None
    if ck['manifest_sha256']!=sha256(cfg.data.manifest):raise ValueError('Manifest mismatch')
    if ds.prepared and (ds.prepared['backbone_checkpoint_sha256']!=backbone or
                       ds.prepared['encoder_source_sha256']!=sha256(Path(__file__).parents[1]/'src/fieldbridge/model.py')):
        raise ValueError('Cached encoder mismatch')
    if cfg.model.feedback=='oracle_shuffled':ds=ShuffledOracleDataset(ds)
    model=build_model(cfg.model).to(cfg.train.device).eval();load_weights(model,ck,backbone)
    gates=[];activations=[];ids=[];mask=None;call=0
    def gate_hook(module,inputs,out):
        nonlocal call
        # Anchored mode also evaluates a zero reference; report real-input pass only.
        if call==0:gates.append(out.float().sigmoid()[mask].cpu().flatten())
        call+=1
    def field_hook(module,inputs,out):
        x=out.float();activations.append([float(x.square().mean().sqrt()),float(x.std())])
    handles=[model.confidence.register_forward_hook(gate_hook),model.field_encoder[0].register_forward_hook(field_hook)]
    try:
        for batch in DataLoader(ds,batch_size=cfg.train.val_batch_size,num_workers=cfg.train.val_workers,collate_fn=collate):
            b=move_batch(batch,torch.device(cfg.train.device));ids.extend(b['sample_id'])
            mask=b['valid'].flatten(1);call=0
            forward_batch(model,b,cfg.model.feedback)
    finally:
        for handle in handles:handle.remove()
    if len(ids)!=len(ds) or len(set(ids))!=len(ds):raise ValueError('Coverage mismatch')
    x=torch.cat(gates);a=torch.tensor(activations)
    result={'samples':len(ids),'split':'val','precision':'fp32','checkpoint':checkpoint,
            'pooling':cfg.model.bridge_pooling,'gate_mean':float(x.mean()),
            'gate_quantiles':dict(zip(['min','p10','p50','p90','max'],x.quantile(torch.tensor([0.,.1,.5,.9,1.])).tolist())),
            'fraction_gate_below_1e4':float((x<1e-4).float().mean()),
            'field_encoder_activation_rms':float(a[:,0].mean()),
            'note':'Sigmoid gates are learned weights, NOT calibrated uncertainty. Legacy divides by frame count; normalized_anchor normalizes gate mass.'}
    atomic_json(output,result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','checkpoint','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();run(a.config,a.checkpoint,a.output)
