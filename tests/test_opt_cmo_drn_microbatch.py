from dataclasses import asdict, replace

import pytest
import torch

from ebl.schema.opt_cmo_drn import Config, parse_config
from methods.programming.cmo import program_rewritten
from networks.opt.physical import CmoDrnBank
from ebl.runtime.opt_cmo_drn.recovery import RecoveryStep
from ebl.runtime.opt_cmo_drn.microbatch_benchmark import gradient_agreement, select_microbatch
from tests.test_opt_cmo_drn import cpu_threads, tiny_model


def test_microbatch_accumulation_matches_same_held_effective_batch():
    model,teacher=tiny_model()
    _,_,state=program_rewritten(model,271002,'random',.03,capture_state=True)
    bank=CmoDrnBank(state,'cpu');before=bank.held_hashes()
    c=Config();c=replace(c,recovery=replace(c.recovery,update_rule='direct_adam',
        target_bounds='gmin_gmax',max_write_attempts=None,write_noise_factor=0.))
    core=RecoveryStep(model,teacher,bank,c)
    core.set_input(torch.randint(0,10,(32,4),generator=torch.Generator().manual_seed(18)))
    reference=tuple(g.clone() for g in core.compute_gradient());loss=core.last['teacher_kl']
    for mb in (4,8,16):
        core.config=replace(c,training=replace(c.training,microbatch=mb))
        assert gradient_agreement(reference,core.compute_gradient())['accepted']
        assert abs(core.last['teacher_kl']-loss)<1e-6
        assert bank.held_hashes()==before
    assert not gradient_agreement(reference,tuple(-g for g in reference))['accepted']


def test_selection_requires_parity_speed_in_both_arms_and_memory():
    parity=[{'microbatch':mb,'accepted':True} for mb in (1,4,8,16) for _ in range(2)]
    timings=[{'arm':arm,'microbatch':mb,'median_step_seconds':seconds,'peak_memory_bytes':6*1024**3}
        for arm in ('noiseless','noisy') for mb,seconds in ((1,2.),(4,1.5),(8,1.2),(16,1.))]
    assert select_microbatch(parity,timings)==16
    timings[-1]['peak_memory_bytes']=29*1024**3
    assert select_microbatch(parity,timings)==8
    parity[4]['accepted']=False
    assert select_microbatch(parity,timings)==4
    timings[5]['median_step_seconds']=1.9
    assert select_microbatch(parity,timings)==1


def test_benchmark_config_is_explicit_and_bounded():
    c=Config();c=replace(c,stage='recovery_benchmark',recovery=replace(c.recovery,
        updates=8,update_rule='direct_adam',target_bounds='gmin_gmax',max_write_attempts=None,write_noise_factor=0.))
    assert parse_config(asdict(c))==c
    with pytest.raises(ValueError,match='benchmark requires'):
        parse_config(asdict(replace(c,training=replace(c.training,effective_batch=16))))
