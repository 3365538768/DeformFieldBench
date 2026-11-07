"""Download an already-authorized HF revision and verify its official Hub adapter.

Access tokens stay in the private HF cache. Run through with_proxy.py on A800.
No access request, license acceptance, or random-weight fallback is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

REPO = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
REVISION = 'ea8dc2863c51be0a264bab82070e3e8836b02d51'
UPSTREAM = '6876159a11b4df116f30f667f8c9888617df0751'


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def restore_original(hf, reference):
    """Invert the official HF converter; only derived buffers may be regenerated."""
    import torch
    converted = {}
    consumed = set()
    for key, template in reference.items():
        if key.endswith('attn.qkv.weight') or key.endswith('attn.qkv.bias'):
            prefix, suffix = key.rsplit('qkv', 1)
            prefix = re.sub(r'blocks\.(\d+)\.attn\.', r'layer.\1.attention.', prefix)
            parts = []
            for letter in 'qkv':
                source = prefix + letter + '_proj' + suffix
                if letter == 'k' and suffix == '.bias' and source not in hf:
                    parts.append(torch.zeros_like(template.chunk(3)[1]))
                else:
                    parts.append(hf[source]); consumed.add(source)
            value = torch.cat(parts)
        elif key.endswith('bias_mask') or key.startswith('rope_embed.'):
            value = template
        else:
            source = key
            for old, new in [('cls_token','embeddings.cls_token'), ('mask_token','embeddings.mask_token'),
                             ('storage_tokens','embeddings.register_tokens'), ('patch_embed.proj','embeddings.patch_embeddings')]:
                if source.startswith(old): source = source.replace(old,new,1); break
            source = re.sub(r'blocks\.(\d+)\.attn\.proj', r'layer.\1.attention.o_proj', source)
            source = re.sub(r'blocks\.(\d+)\.ls(\d+)\.gamma', r'layer.\1.layer_scale\2.lambda1', source)
            source = re.sub(r'blocks\.(\d+)\.mlp\.fc1', r'layer.\1.mlp.up_proj', source)
            source = re.sub(r'blocks\.(\d+)\.mlp\.fc2', r'layer.\1.mlp.down_proj', source)
            source = re.sub(r'blocks\.(\d+)\.norm', r'layer.\1.norm', source)
            # The global/local class norms are tied for the LVD model at inference.
            if source.startswith('local_cls_norm.'): source = source.replace('local_cls_norm.', 'norm.')
            value = hf[source]; consumed.add(source)
            if key == 'mask_token': value = value.squeeze(1)
        if value.shape != template.shape: raise ValueError('Shape mismatch: ' + key)
        converted[key] = value.to(template.dtype)
    if set(hf) != consumed: raise ValueError('Unconsumed pretrained tensors: ' + str(set(hf)-consumed))
    return converted


def prepare(status):
    status = Path(status)
    import subprocess
    actual = subprocess.check_output(['git','-C','external/dinov3','rev-parse','HEAD'],text=True).strip()
    if actual!=UPSTREAM: raise ValueError('Upstream code revision mismatch')
    if status.exists():
        previous=json.loads(status.read_text())
        weights=Path(previous.get('weights','__missing__'))
        if (previous.get('status')=='complete' and previous.get('revision')==REVISION
                and previous.get('upstream_commit')==UPSTREAM and previous.get('strict_load')
                and weights.is_file() and digest(weights)==previous.get('weights_sha256')):
            print(json.dumps(previous,indent=2)); return
    started = time.time(); directory = Path('weights/dinov3_hf'); directory.mkdir(parents=True, exist_ok=True)
    hf_home=Path(os.environ.get('HF_HOME',Path.home()/'.cache/huggingface'))
    token = Path(os.environ.get('HF_TOKEN_PATH',hf_home/'token')).read_text().strip()
    def request(url, headers=None):
        return urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization':'Bearer '+token, **(headers or {})}), timeout=120)
    api = json.load(request('https://huggingface.co/api/models/'+REPO+'/revision/'+REVISION+'?blobs=true'))
    model_info = next(s for s in api['siblings'] if s['rfilename']=='model.safetensors')
    expected = model_info['lfs']['sha256']; size = model_info['size']
    base = 'https://huggingface.co/'+REPO+'/resolve/'+REVISION+'/'
    for name in ('config.json','preprocessor_config.json','LICENSE.md','model.safetensors'):
        path = directory/name
        if path.exists() and (name!='model.safetensors' or digest(path)==expected): continue
        temp = path.with_suffix(path.suffix+'.part'); offset = temp.stat().st_size if temp.exists() else 0
        with request(base+name, {'Range':f'bytes={offset}-'} if offset else {}) as response:
            if offset and response.status!=206: raise ValueError('Server did not honor resume Range')
            with temp.open('ab' if offset else 'wb') as output:
                while block := response.read(8*1024*1024):
                    output.write(block); offset += len(block)
                    atomic(status, dict(status='downloading',file=name,completed_bytes=offset,total_bytes=size,
                        repository=REPO,revision=REVISION,updated_unix=time.time()))
        if name=='model.safetensors' and (offset!=size or digest(temp)!=expected): raise ValueError('HF LFS SHA256 mismatch')
        temp.replace(path)
    atomic(status, dict(status='converting',repository=REPO,revision=REVISION,updated_unix=time.time()))
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel
    original = torch.hub.load('external/dinov3','dinov3_vitl16',source='local',pretrained=False).eval()
    original.load_state_dict(restore_original(load_file(str(directory/'model.safetensors')), original.state_dict()),strict=True)
    hf = AutoModel.from_pretrained(directory,local_files_only=True).eval()
    original = original.cuda(); hf = hf.cuda()
    torch.manual_seed(17); x=torch.randn(2,3,224,224,device='cuda')
    with torch.inference_mode():
        a=original.forward_features(x)['x_norm_patchtokens']; b=hf(x).last_hidden_state[:,5:]
    maximum=float((a-b).abs().max()); mean=float((a-b).abs().mean())
    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
    temporary=Path('weights/dinov3_vitl16_import.part');torch.save(original.cpu().state_dict(),temporary)
    checksum=digest(temporary); target=Path('weights/dinov3_vitl16_pretrain_lvd1689m-'+checksum[:8]+'.pth')
    temporary.replace(target)
    atomic(status,dict(status='complete',repository=REPO,revision=REVISION,upstream_commit=UPSTREAM,
        source_sha256=expected,weights=str(target),weights_sha256=checksum,strict_load=True,
        fp32_patch_max_delta=maximum,fp32_patch_mean_delta=mean,parity_images=2,
        feature_shape=list(a.shape),elapsed_seconds=time.time()-started,updated_unix=time.time()))
    print(json.dumps(json.loads(status.read_text()),indent=2))


if __name__=='__main__':
    p = argparse.ArgumentParser(); p.add_argument('--status', default='docs/DINOV3_PREPARATION.json')
    args = p.parse_args()
    try:
        prepare(args.status)
    except Exception as error:
        # Exception text can contain redirected signed URLs; expose only its class.
        atomic(args.status, dict(status='failed',error_type=type(error).__name__,
            repository=REPO,revision=REVISION,updated_unix=time.time()))
        raise SystemExit('DINOv3 preparation failed: '+type(error).__name__)
