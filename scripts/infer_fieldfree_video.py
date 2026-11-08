"""A single RGB video -> native material estimates, without any field inputs."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fieldbridge.config import ModelConfig
from fieldbridge.data import ParameterCodec,PARAMETERS
from fieldbridge.local_model import build_model
from fieldbridge.paths import path_map,relocate_path
from fieldbridge.posterior import mixture_quantile
from fieldbridge.runtime import atomic_json,load_weights,sha256
from prepare_fieldfree_priors import ridge_predict

ROOT=Path(__file__).resolve().parents[1]


def read_video(path,frames,image_size,timestamps=None):
    import cv2
    capture=cv2.VideoCapture(str(path))
    try:
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or total<1:
            raise ValueError('Video needs decodable RGB frames')
        if timestamps is not None:
            clock=torch.tensor(json.loads(Path(timestamps).read_text()),dtype=torch.float64)
            if (clock.shape!=(total,) or not torch.isfinite(clock).all()
                    or not (clock[1:]>clock[:-1]).all()):
                raise ValueError('Timestamps need one finite, strictly increasing value in seconds per source frame')
            origin=float(clock[0]);clock=clock-origin;timing_source='explicit_timestamps_seconds'
        else:
            if not np.isfinite(fps) or fps<=0:
                raise ValueError('Video needs valid FPS or explicit timestamps; no guessed physical time')
            origin=0.;clock=torch.arange(total,dtype=torch.float64)/fps;timing_source='constant_container_fps'
        indices=torch.linspace(0,total-1,min(frames,total)).round().long().tolist();images=[]
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=capture.read()
            if not ok:raise ValueError(f'Failed to decode video frame {index}')
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None]/255.
            images.append(torch.nn.functional.interpolate(tensor,(image_size,image_size),mode='bilinear',align_corners=False)[0])
        sampled_times=clock[indices].float()
        return torch.stack(images),sampled_times,dict(source_frames=total,fps=fps if np.isfinite(fps) and fps>0 else None,
            sampled_frame_indices=indices,sampled_timestamps_seconds=sampled_times.tolist(),timing_source=timing_source,
            source_time_origin_seconds=origin)
    finally:
        capture.release()


def run(checkpoint,video,output,device='cuda',timestamps=None):
    checkpoint,video,output=[Path(p).resolve() for p in [checkpoint,video,output]]
    timestamps=Path(timestamps).resolve() if timestamps is not None else None
    os.chdir(ROOT);torch.set_num_threads(1);started=time.perf_counter()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);raw=dict(state['config']['model'])
    for key in ['repo','weights']:raw[key]=relocate_path(raw[key],path_map())
    cfg=ModelConfig(**raw)
    if cfg.architecture!='direct_video' or cfg.field_mode!='none' or cfg.feedback!='none':
        raise ValueError('Requires an explicitly field-free checkpoint')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    data=state['config']['data'];model=build_model(cfg).to(device).eval();load_weights(model,state,sha256(cfg.weights) if cfg.weights else None)
    rgb,times,metadata=read_video(video,data['frames'],data['image_size'],timestamps)
    rgb=rgb[None,None].to(device);times=times[None,None].to(device);valid=torch.ones_like(times,dtype=torch.bool)
    prior=torch.zeros(1,4,device=device);mask=torch.zeros(1,4,dtype=torch.bool,device=device);lv=torch.zeros(1,4,device=device)
    with torch.no_grad(),torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
        features=model.frame_encoder(rgb.flatten(0,2)).reshape(1,1,len(metadata['sampled_frame_indices']),model.frame_encoder.dim,-1)
        side=int(features.shape[-1]**.5)
        if side*side!=features.shape[-1]:raise ValueError('Expected square patch grid')
        features=features.reshape(1,1,features.shape[2],features.shape[3],side,side)
        if cfg.use_prior:
            prior_path=Path(relocate_path(data['material_prior_file'],path_map()))
            if sha256(prior_path)!=state['material_overlay_sha256']['material_prior_file']:
                raise ValueError('Static prior artifact changed since training')
            index=json.loads(prior_path.read_text());entry=index['fits']['full_train'];fit_path=ROOT/entry['checkpoint']
            if sha256(fit_path)!=entry['checkpoint_sha256']:raise ValueError('Static fit checkpoint SHA mismatch')
            fit=torch.load(fit_path,map_location='cpu',weights_only=True)
            # Match prior generation FP32 math, outside the video BF16 autocast.
            with torch.autocast(device_type=device,enabled=False):
                estimate=ridge_predict(fit,features[0,0,0].float().mean((-1,-2))[None].cpu())
            prior=estimate.to(device);mask=fit['mask'][None].to(device);lv=fit['logvar'][None].to(device)
        prediction=model(rgb=rgb,valid=valid,times=times,prior=prior,prior_mask=mask,prior_logvar=lv,features=features)
    codec=ParameterCodec(state['stats']);mean=codec.decode(prediction['mean'].float().cpu())[0]
    components=[prediction[k].float().cpu() for k in ['mixture_logits','mixture_means','mixture_logvars']]
    trained_uncertainty=state['config']['train']['nll_weight']>0
    intervals=None
    if trained_uncertainty:
        lower=codec.decode(mixture_quantile(*components,.025).float())[0]
        upper=codec.decode(mixture_quantile(*components,.975).float())[0]
        intervals={name:[float(lower[j]),float(upper[j])] for j,name in enumerate(PARAMETERS)}
    result=dict(status='complete',input_video_sha256=sha256(video),input_timestamps_sha256=sha256(timestamps) if timestamps else None,
        checkpoint_sha256=sha256(checkpoint),
        parameters={name:float(mean[j]) for j,name in enumerate(PARAMETERS)},native_marginal_95_intervals=intervals,
        uncertainty_trained=trained_uncertainty,law_probabilities=prediction['law_logits'].float().softmax(-1)[0].cpu().tolist(),
        law_order=['jelly','metal','plasticine'],video=metadata,elapsed_including_setup_seconds=time.perf_counter()-started,
        uses_fields=False,uses_online_optimization=False,
        scope='Original training-domain effective targets. Real-video accuracy, loading/geometry identifiability and cross-engine conversion are not established. Untrained uncertainty is omitted; trained intervals are uncalibrated.')
    atomic_json(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--video',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--timestamps-json',help='Optional array of physical seconds, one per source frame; required for irregularly sampled exports')
    args=parser.parse_args()
    print(json.dumps(run(args.checkpoint,args.video,args.output,args.device,args.timestamps_json),ensure_ascii=False,indent=2))
