import copy
import pytest
import torch
from fieldbridge.engine_domains import EngineDomain, EngineConditionedHeads, require_matched_response, validate_single_parameter_domain


def domain(version='corrected'):
    return {'engine':'libuipc','version':'0.0.29','constitutive_law':'SNK1',
            'coefficient_mapping':version,'solver_protocol':'beam-calibration-v1',
            'parameter_units':{'E':'Pa','nu':'1'}}


def specification(value):
    return {'domain':value,'parameters':['E','nu'],'transforms':['log1p','linear'],
            'mean':[10.,.3],'std':[1.,.1],'fit_split':'train','fit_manifest_sha256':'train-manifest'}


def test_native_labels_cannot_silently_share_a_codec():
    first={'source':'original','parameters':{'E':1e5,'nu':.3},'engine_domain':domain()}
    second=copy.deepcopy(first);second['engine_domain']=domain('old_native')
    assert validate_single_parameter_domain([first])==EngineDomain.parse(domain()).key
    with pytest.raises(ValueError,match='Multiple native'):validate_single_parameter_domain([first,second])
    legacy={'source':'original','parameters':{'E':1e5}}
    assert validate_single_parameter_domain([legacy])=='legacy_source:original'
    with pytest.raises(ValueError,match='Multiple native'):validate_single_parameter_domain([first,legacy])
    with pytest.raises(ValueError,match='Multiple native'):validate_single_parameter_domain([legacy,{**legacy,'source':'other'}])
    assert validate_single_parameter_domain([first,{'source':'real','parameters':{}}])==EngineDomain.parse(domain()).key
    bad=copy.deepcopy(first);bad['parameters']['rho']=1000
    with pytest.raises(ValueError,match='unit'):validate_single_parameter_domain([bad])


def test_engine_head_routing_has_no_effect_on_shared_response_and_checks_checkpoint():
    torch.manual_seed(17)
    a,b=domain(),domain('old_native');ka,kb=EngineDomain.parse(a).key,EngineDomain.parse(b).key
    specs={ka:specification(a),kb:specification(b)}
    model=EngineConditionedHeads(8,specs,['tip_compliance_m_per_n','rebound_ratio'])
    latent=torch.randn(3,8,requires_grad=True)
    result=model(latent,[ka,kb,ka]);swapped=model(latent,[kb,ka,kb])
    torch.testing.assert_close(result['response'],swapped['response'],rtol=0,atol=0)
    assert result['native'][ka]['indices'].tolist()==[0,2]
    assert result['native'][kb]['indices'].tolist()==[1]
    (result['response'].sum()+sum(v['mean_z'].sum() for v in result['native'].values())).backward()
    assert torch.isfinite(latent.grad).all() and all(h.weight.grad.abs().sum()>0 for h in model.heads.values())
    EngineConditionedHeads(8,specs,model.response_names).load_state_dict(model.state_dict())
    changed=copy.deepcopy(specs);changed[ka]['std'][0]=2
    with pytest.raises(ValueError,match='checkpoint'):EngineConditionedHeads(8,changed,model.response_names).load_state_dict(model.state_dict())
    with pytest.raises(ValueError,match='Unknown engine'):model(latent,['unseen']*3)


def test_relative_response_requires_matched_scene_and_observable():
    fixture={'protocol_sha256':'includes-strain-rate-temperature','fixture_geometry_sha256':'mesh',
             'length_scale_m':.2,'load_history_sha256':'force','boundary_sha256':'clamp',
             'observable':'tangent_tip_compliance','units':'m/N'}
    assert require_matched_response(fixture,fixture)
    for key,value in [('length_scale_m',.1),('load_history_sha256','other-force'),('units','N/m')]:
        with pytest.raises(ValueError,match='Cannot rank'):require_matched_response(fixture,{**fixture,key:value})
    with pytest.raises(ValueError,match='Missing'):require_matched_response({},fixture)
    bad={**fixture,'length_scale_m':float('nan')}
    with pytest.raises(ValueError):require_matched_response(bad,bad)
