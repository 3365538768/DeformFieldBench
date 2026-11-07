"""Explicit native-label domains and experimental shared-response/engine heads.

Equal parameter names do not establish physical equivalence. The current
single-codec trainer rejects heterogeneous native domains; the adapter below is
an independently tested building block, not an integrated multi-engine trainer.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn


@dataclass(frozen=True)
class EngineDomain:
    engine: str
    version: str
    constitutive_law: str
    coefficient_mapping: str
    solver_protocol: str
    parameter_units: tuple

    @classmethod
    def parse(cls, value):
        required={'engine','version','constitutive_law','coefficient_mapping','solver_protocol','parameter_units'}
        if not isinstance(value,dict) or set(value)!=required:
            raise ValueError('Engine domain needs engine/version/law/mapping/solver/parameter_units')
        for key in required-{'parameter_units'}:
            if not isinstance(value[key],str) or not value[key].strip():
                raise ValueError('Empty engine domain identity: '+key)
        units=value['parameter_units']
        if not isinstance(units,dict) or not units or any(not isinstance(k,str) or not isinstance(v,str) or not v for k,v in units.items()):
            raise ValueError('Native parameter units must be explicit')
        return cls(**{k:value[k] for k in required-{'parameter_units'}},parameter_units=tuple(sorted(units.items())))

    @property
    def key(self):
        content=json.dumps(asdict(self),sort_keys=True,separators=(',',':'))
        return 'engine_'+hashlib.sha256(content.encode()).hexdigest()[:24]


def validate_single_parameter_domain(rows):
    """Fail closed when the existing one-head/one-codec loader would mix labels.

    Historical unannotated rows from ONE source remain a legacy domain. This
    preserves existing manifests without asserting that their parameters are SI.
    Unlabelled videos can be inventoried but do not create parameter supervision.
    """
    keys=set()
    for row in rows:
        parameters=row.get('parameters',{})
        if not any(v is not None for v in parameters.values()):
            continue
        raw=row.get('engine_domain')
        if raw is None:
            keys.add('legacy_source:'+str(row['source']))
            continue
        domain=EngineDomain.parse(raw)
        units=dict(domain.parameter_units)
        if any(name not in units for name,value in parameters.items() if value is not None):
            raise ValueError('Label missing native parameter unit')
        keys.add(domain.key)
    if len(keys)>1:
        raise ValueError('Multiple native parameter domains require engine-specific heads/codecs; single-domain trainer refuses mixed GT')
    return next(iter(keys),None)


class EngineConditionedHeads(nn.Module):
    """Prototype: shared response embedding plus separate native parameter heads.

    Caller must provide train-only normalization per domain and explicit masked
    supervision. Engine ID selects a native head; it is NOT an input to the shared
    response head. Unknown engines are rejected instead of using another head.
    Native units/statistics are serialized and verified with the state dict.
    """
    def __init__(self,width,specifications,response_names):
        super().__init__()
        if not specifications or not response_names or len(set(response_names))!=len(response_names):
            raise ValueError('Need native domains and distinct response names')
        self.specifications={}
        self.heads=nn.ModuleDict()
        for key,spec in specifications.items():
            domain=EngineDomain.parse(spec['domain'])
            if domain.key!=key:
                raise ValueError('Domain key mismatch')
            names=spec['parameters'];size=len(names)
            if not size or len(set(names))!=size or set(names)!=set(dict(domain.parameter_units)):
                raise ValueError('Parameter schema/units mismatch')
            if spec.get('fit_split')!='train' or not spec.get('fit_manifest_sha256'):
                raise ValueError('Domain normalization must have train-only provenance')
            if len(spec['mean'])!=size or len(spec['std'])!=size or any(not math.isfinite(v) for v in spec['mean']) or any(not math.isfinite(v) or v<=0 for v in spec['std']):
                raise ValueError('Invalid domain normalization')
            if len(spec['transforms'])!=size or any(t not in ('linear','log1p') for t in spec['transforms']):
                raise ValueError('Explicit native transforms required')
            self.specifications[key]=json.loads(json.dumps(spec))
            self.heads[key]=nn.Linear(width,2*size)
        self.response_names=tuple(response_names)
        self.shared_response=nn.Linear(width,len(response_names))

    def get_extra_state(self):
        return {'specifications':self.specifications,'response_names':self.response_names}

    def set_extra_state(self,state):
        # Reject a checkpoint with changed units, calibration, or normalization.
        if state!=self.get_extra_state():
            raise ValueError('Adapter checkpoint domain/schema/normalization mismatch')

    def forward(self,latent,engine_ids):
        if latent.ndim!=2 or len(engine_ids)!=len(latent):
            raise ValueError('One native domain per sample required')
        if any(key not in self.heads for key in engine_ids):
            raise ValueError('Unknown engine domain; calibrated adapter required')
        native={}
        for key in sorted(set(engine_ids)):
            indices=torch.tensor([i for i,k in enumerate(engine_ids) if k==key],device=latent.device)
            mean,logvar=self.heads[key](latent.index_select(0,indices)).chunk(2,-1)
            native[key]={'indices':indices,'mean_z':mean,'logvar_z':logvar.clamp(-6,4),
                         'parameters':tuple(self.specifications[key]['parameters'])}
        return {'native':native,'response':self.shared_response(latent),'response_names':self.response_names}


RESPONSE_MATCH_KEYS=('protocol_sha256','fixture_geometry_sha256','length_scale_m',
                     'load_history_sha256','boundary_sha256','observable','units')


def require_matched_response(a,b):
    """Relative supervision is valid only inside the same controlled response test.

    Protocol hash must include operating strain/rate/temperature and extraction
    conventions. A successful check is necessary, not proof of engine equivalence.
    """
    if any(key not in a or key not in b for key in RESPONSE_MATCH_KEYS):
        raise ValueError('Missing response matching provenance')
    if any(a[key]!=b[key] for key in RESPONSE_MATCH_KEYS):
        raise ValueError('Cannot rank unmatched geometry/load/scale/boundary/observable')
    if any(a[key] in ('',None) for key in RESPONSE_MATCH_KEYS):
        raise ValueError('Empty response matching provenance')
    if not isinstance(a['length_scale_m'],(int,float)) or not math.isfinite(a['length_scale_m']) or a['length_scale_m']<=0:
        raise ValueError('Response scale must be positive SI metres')
    return True
