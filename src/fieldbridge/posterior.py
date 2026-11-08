"""Exact marginal mixture intervals; never use moment-Gaussian coverage as exact."""
import math
import torch
from .direct_video import mixture_log_prob


def mixture_quantile(logits, means, logvars, probability):
    if not 0 < probability < 1:
        raise ValueError('Quantile probability must be interior')
    weights=logits.double().softmax(-1)[...,None]
    means=means.double();sigma=(.5*logvars.double()).exp()
    lower=(means-12*sigma).amin(-2);upper=(means+12*sigma).amax(-2)
    for _ in range(48):
        midpoint=(lower+upper)/2
        cdf=(weights*.5*(1+torch.erf((midpoint[...,None,:]-means)/(math.sqrt(2)*sigma)))).sum(-2)
        lower=torch.where(cdf<probability,midpoint,lower)
        upper=torch.where(cdf>=probability,midpoint,upper)
    return (lower+upper)/2


def posterior_summary(records):
    logits,means,logvars=[torch.tensor([r[k] for r in records],dtype=torch.float64) for k in
                          ['mixture_logits','mixture_means','mixture_logvars']]
    target=torch.tensor([r['target_z'] for r in records],dtype=torch.float64)
    mask=torch.tensor([r['target_mask'] for r in records],dtype=torch.bool)
    lower=mixture_quantile(logits,means,logvars,.025);upper=mixture_quantile(logits,means,logvars,.975)
    count=mask.sum(-1);keep=count>0
    nll=-mixture_log_prob(logits,means,logvars,target,mask)
    coverage=((target>=lower)&(target<=upper)).double()
    return dict(joint_nll_per_observed_parameter=float(nll[keep].sum()/count[keep].sum()),
        marginal_95_coverage=float(coverage[mask].mean()),mean_interval_width_z=float((upper-lower)[mask].mean()),
        per_parameter=[dict(count=int(mask[:,j].sum()),coverage=float(coverage[:,j][mask[:,j]].mean()) if mask[:,j].any() else None,
                            mean_interval_width_z=float((upper-lower)[:,j][mask[:,j]].mean()) if mask[:,j].any() else None) for j in range(4)],
        interval_method='Numerical inversion of the exact marginal mixture CDF; empirical validation coverage, no calibration guarantee.')
