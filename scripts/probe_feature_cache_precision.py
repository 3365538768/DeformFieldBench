"""Bounded candidate-cache probe; does not replace any experiment input cache."""
import json
import os
from pathlib import Path
import time
import torch
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset,collate,model_inputs
from fieldbridge.local_model import build_model
from fieldbridge.runtime import load_weights,sha256
from engine_fixture import atomic,wait_if_paused

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);torch.set_num_threads(1);out=ROOT/'runs/cache_precision_probe';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=16,rows=[],test_untouched=True,
        detail='16条固定验证样本：原FP32、FP16、BF16特征与冗余旧场剥离候选；只做受控预测parity，不替换训练缓存。',
        scope='Bounded validation-selected checkpoint diagnostic. Warm file loading is not full training or cold shared-disk throughput. Candidates are not admitted for training.')
    try:
        cfg=load_config('configs/runtime/m3_D_raw_seed17.json');ds=MaterialDataset(cfg.data,'val',cfg.model.field_size,cfg.model.field_mode)
        model=build_model(cfg.model).cuda().eval();ckpt=torch.load(ROOT/'runs/m3_D_raw_seed17/best_material.pt',map_location='cpu',weights_only=False)
        load_weights(model,ckpt,sha256(cfg.model.weights));base=ROOT/cfg.data.prepared_dir
        bytes_by_mode=dict(original=0,fp16=0,bf16=0);max_delta=dict(fp16=0.,bf16=0.);max_feature_error=dict(fp16=0.,bf16=0.)
        for i in range(16):
            wait_if_paused(out);sample=ds[i];sid=sample['sample_id'];original_path=base/ds.prepared['files'][sid]
            original=torch.load(original_path,map_location='cpu',weights_only=True);bytes_by_mode['original']+=original_path.stat().st_size
            def predict(record):
                inputs={k:v.cuda() if torch.is_tensor(v) else v for k,v in model_inputs(collate([record])).items()}
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model(**inputs)['mean'].float().cpu()
            target=predict(sample)
            row=dict(sample_id=sid,original_sha256=sha256(original_path),prediction_max_z_delta={})
            for mode,dtype in [('fp16',torch.float16),('bf16',torch.bfloat16)]:
                candidate=dict(original);candidate.pop('fields',None);candidate.pop('field_mask',None)
                candidate['features']=original['features'].to(dtype);path=out/(str(i)+'_'+mode+'.pt');torch.save(candidate,path)
                bytes_by_mode[mode]+=path.stat().st_size
                changed=dict(sample,features=sample['features'].to(dtype))
                delta=float((predict(changed)-target).abs().max());max_delta[mode]=max(max_delta[mode],delta)
                feature_delta=float((candidate['features'].float()-original['features']).abs().max());max_feature_error[mode]=max(max_feature_error[mode],feature_delta)
                row['prediction_max_z_delta'][mode]=delta;row[mode+'_sha256']=sha256(path)
                for key in ['valid','times','target','target_mask','prior','prior_mask']:
                    if not torch.equal(original[key],candidate[key]):raise ValueError('Candidate changed label/time/mask')
            report['rows'].append(row);report.update(completed=i+1,updated_unix=time.time());atomic(ROOT/'docs/CACHE_PRECISION_PROBE.json',report)
        loading={}
        for mode in ['original','fp16','bf16']:
            passes=[]
            for _ in range(2):
                wait_if_paused(out);start=time.perf_counter()
                for i in range(16):
                    path=base/ds.prepared['files'][ds.rows[i]['sample_id']] if mode=='original' else out/(str(i)+'_'+mode+'.pt')
                    record=torch.load(path,map_location='cpu',weights_only=True)
                    if not torch.isfinite(record['features']).all():raise ValueError('Non-finite candidate feature')
                passes.append(time.perf_counter()-start)
            loading[mode]=passes
        report.update(status='complete',updated_unix=time.time(),bytes_by_mode=bytes_by_mode,
            prediction_max_z_delta=max_delta,feature_max_absolute_error=max_feature_error,
            passes_point_prediction_gate={k:v<=.001 for k,v in max_delta.items()},point_prediction_gate=.001,
            serial_warm_load_and_finite_check_seconds=loading,
            next_step='Only passing precision may proceed to full 128-sample outputs/uncertainty and resume parity, explicit cache-format contract, then matched training-time benchmark. Existing caches remain fixed.')
        atomic(ROOT/'docs/CACHE_PRECISION_PROBE.json',report);atomic(out/'status.json',dict(status='complete',reason='bounded_precision_probe_complete_not_training_admission'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/CACHE_PRECISION_PROBE.json',report);atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
