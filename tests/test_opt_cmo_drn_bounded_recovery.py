"""Bounded-target direct writes, controller parity and policy separation."""

from dataclasses import replace

import pytest
import torch

from experiments.opt_cmo_drn.config import Config, Recovery, parse_config
from experiments.opt_cmo_drn.physical import ConductanceArray
from experiments.opt_cmo_drn.recovery import RecoveryStep, RecoveryWritePort
from training.cmo_direct_adam import DirectWriteAdam, ThresholdedBoundedWriteAdam
from training.reram_cmo import G_MIN_US, G_MAX_US, HALF_RANGE_US, programming_sigma
from tests.test_opt_cmo_drn import cpu_threads, tiny_model


def array_at(values, stuck=None):
    held=torch.tensor(values,dtype=torch.float32)
    mask=torch.zeros_like(held,dtype=torch.bool)
    if stuck is not None:mask[stuck]=True
    return ConductanceArray({'name':'test','noise_scale':1.,'held':held,'target':held,
        'mask':mask,'stuck':held.clone(),'raw':held,'verify_flags':mask,
        'rewrite_selected':torch.zeros_like(mask),'rng':torch.Generator().manual_seed(9).get_state(),
        'write_attempts':0,'device_writes':0,'stuck_write_attempts':0},'cpu')


@pytest.mark.parametrize('noise_factor',[0.,1.])
def test_project_targets_before_noise_preserve_faults_and_untouched_cells(noise_factor):
    array=array_at([40.,40.,G_MIN_US,G_MAX_US,6.,40.],stuck=5)
    before=array.held.clone()
    command=torch.tensor([-10.,10.,-1.,1.,0.,.1])
    target=(before+HALF_RANGE_US*command).clamp(G_MIN_US,G_MAX_US)
    rng=torch.Generator();rng.set_state(array.generator.get_state())
    draws=torch.randn(command.shape,generator=rng)
    expected=target if noise_factor==0 else target+programming_sigma(target,2.)*draws
    attempted=array.write(command,noise_factor=noise_factor,target_bounds='gmin_gmax')
    assert torch.equal(attempted,torch.tensor([True,True,False,False,False,True]))
    assert torch.equal(array.held[:2],expected[:2])
    assert torch.equal(array.held[2:],before[2:])
    assert torch.equal(array.generator.get_state(),rng.get_state())
    assert array.counters['write_attempts']==3 and array.counters['stuck_write_attempts']==1
    assert array.bounded_counters=={'proposed_commands':5,'target_clipped_low':2,
        'target_clipped_high':2,'projected_noops':2,'state_changing_writes':2}


def test_noisy_realized_endpoints_are_not_clipped_and_noops_do_not_redraw_state():
    array=array_at([40.]*128)
    array.write(torch.full_like(array.held,-10.),target_bounds='gmin_gmax')
    assert bool((array.held<G_MIN_US).any())
    array.write(torch.full_like(array.held,10.),target_bounds='gmin_gmax')
    assert bool((array.held>G_MAX_US).any())
    before=array.held.clone();rng=array.generator.get_state().clone()
    array.write(torch.zeros_like(array.held),target_bounds='gmin_gmax')
    assert torch.equal(array.held,before)
    assert not torch.equal(array.generator.get_state(),rng)


def test_direct_adam_matches_torch_adam_commands_without_accumulation():
    class Port:
        def read(self):return torch.zeros(3)
        def write(self,command):
            self.command=command.clone()
            return torch.tensor([True,False,True])
    port=Port();writer=DirectWriteAdam(port,learning_rate=.003)
    parameter=torch.zeros(3,requires_grad=True)
    reference=torch.optim.Adam([parameter],lr=.003,betas=(.9,.999),eps=1e-8,foreach=False)
    for gradient in (torch.tensor([1.,-.5,0.]),torch.tensor([-.2,.7,.4]),torch.zeros(3)):
        previous=parameter.detach().clone()
        parameter.grad=gradient.clone();reference.step();writer.step(gradient)
        torch.testing.assert_close(port.command,parameter.detach()-previous,atol=1e-9,rtol=1e-5)
    assert torch.equal(writer.attempt_counts,torch.tensor([3,0,3]))
    assert not hasattr(writer,'pending')
    clone=DirectWriteAdam(port,learning_rate=.003);clone.load_state_dict(writer.state_dict())
    with pytest.raises(ValueError,match='contract'):
        DirectWriteAdam(port,learning_rate=.001).load_state_dict(writer.state_dict())


def test_bounded_write_resume_and_noise_arms_share_initial_state_and_rng():
    source=array_at([9.,40.,88.199997,20.],stuck=1).state_dict()
    arms=[ConductanceArray(source,'cpu') for _ in range(2)]
    writers=[DirectWriteAdam(RecoveryWritePort(a,f,target_bounds='gmin_gmax'),learning_rate=.003)
             for a,f in zip(arms,(0.,1.))]
    assert torch.equal(arms[0].held,arms[1].held)
    for writer in writers:writer.step(torch.tensor([1.,-1.,-.5,.7]))
    assert torch.equal(arms[0].generator.get_state(),arms[1].generator.get_state())
    assert arms[0].held[1]==arms[1].held[1]==40.
    for array,writer,factor in zip(arms,writers,(0.,1.)):
        clone=ConductanceArray(array.state_dict(),'cpu')
        resumed=DirectWriteAdam(RecoveryWritePort(clone,factor,target_bounds='gmin_gmax'),learning_rate=.003)
        resumed.load_state_dict(writer.state_dict())
        for _ in range(6):
            gradient=torch.tensor([-.2,.3,.4,-.6]);writer.step(gradient);resumed.step(gradient)
        assert torch.equal(array.held,clone.held)
        assert torch.equal(array.generator.get_state(),clone.generator.get_state())
        assert array.counters==clone.counters and array.bounded_counters==clone.bounded_counters
        assert int(writer.attempt_counts.sum())==array.counters['write_attempts']
        assert writer.attempt_counts.max()>4


def test_dense_write_diagnostic_capture_is_bounded():
    array=array_at([40.]*10000)
    port=RecoveryWritePort(array,0.,capture=True,target_bounds='gmin_gmax')
    port.write(torch.ones_like(array.held)*.001)
    assert port.record['indices'].numel()==64
    assert port.record['proposed_commands']==10000


def test_direct_policy_is_explicit_and_rejects_realized_clipping_or_write_caps():
    good={'update_rule':'direct_adam','target_bounds':'gmin_gmax','max_write_attempts':None}
    assert parse_config({'recovery':good}).recovery.update_rule=='direct_adam'
    for change in ({'target_bounds':'none'},{'max_write_attempts':4},{'conductance_floor':'gmin'},
                   {'update_rule':'thresholded_adam'},{'update_rule':'typo'}):
        with pytest.raises(ValueError):parse_config({'recovery':{**good,**change}})


def test_direct_gradients_use_held_state_and_digital_parameters_are_frozen():
    from experiments.opt_cmo_drn.devices import program_rewritten
    from experiments.opt_cmo_drn.physical import CmoDrnBank
    model,teacher=tiny_model()
    _,_,state=program_rewritten(model,271002,'random',.03,capture_state=True)
    bank=CmoDrnBank(state,'cpu')
    config=replace(Config(),recovery=replace(Recovery(),update_rule='direct_adam',
        target_bounds='gmin_gmax',max_write_attempts=None,write_noise_factor=0.))
    step=RecoveryStep(model,teacher,bank,config)
    step.set_input(torch.tensor([[1,2,3,4],[4,3,2,1]]))
    gradients=tuple(g.clone() for g in step.compute_gradient())
    with torch.no_grad():
        for p in model.parameter_groups()[0]:p.add_(.123)
    again=step.compute_gradient()
    for a,b in zip(gradients,again):assert torch.equal(a,b)
    before={k:v.clone() for k,v in model.state_dict().items()}
    step.step()
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    assert all(not p.requires_grad for group in model.parameter_groups()[1:] for p in group)


def test_noisy_gate_uses_each_bounded_target_write_standard_deviation():
    array=array_at([20.,70.,40.],stuck=2)
    port=RecoveryWritePort(array,1.,target_bounds='gmin_gmax')
    writer=ThresholdedBoundedWriteAdam(port,learning_rate=.001)
    writer.pending.copy_(torch.tensor([.02,.02,.1]))
    target=(array.held+HALF_RANGE_US*writer.pending).clamp(G_MIN_US,G_MAX_US)
    assert torch.equal(port.write_noise_std_q(writer.pending),programming_sigma(target,2.)/HALF_RANGE_US)
    before=array.held.clone()
    writer.step(torch.zeros(3))
    assert torch.equal(writer.attempt_counts,torch.tensor([1,0,1]))
    assert array.held[1]==before[1] and array.held[2]==before[2]
    assert torch.equal(writer.pending,torch.tensor([0.,.02,0.]))
    assert array.counters['stuck_write_attempts']==1
    assert writer.diagnostics()['threshold_reference']=='bounded_target_write_std'
    saved=writer.state_dict()
    clone=ThresholdedBoundedWriteAdam(port,learning_rate=.001);clone.load_state_dict(saved)
    assert torch.equal(clone.pending,writer.pending)


def test_two_rho_gate_is_inclusive_and_signed_residuals_are_not_flushed():
    class Port:
        def read(self):return torch.zeros(3)
        def write_noise_std_q(self,command):return torch.full_like(command,.01)
        def write(self,command):self.command=command.clone();return command!=0
    port=Port();writer=ThresholdedBoundedWriteAdam(port,learning_rate=.001)
    writer.pending.copy_(torch.tensor([.02,-.02,.019]))
    writer.step(torch.zeros(3))
    assert torch.equal(port.command,torch.tensor([.02,-.02,0.]))
    assert torch.equal(writer.pending,torch.tensor([0.,0.,.019]))
    assert torch.equal(writer.state_dict()['pending'],writer.pending)


def test_bounded_noisy_gate_consumes_outward_saturated_command_without_a_write():
    array=array_at([G_MAX_US,G_MIN_US])
    writer=ThresholdedBoundedWriteAdam(RecoveryWritePort(array,1.,target_bounds='gmin_gmax'),learning_rate=.001)
    writer.pending.copy_(torch.tensor([.1,-.1]))
    before=array.held.clone();writer.step(torch.zeros(2))
    assert torch.equal(array.held,before)
    assert not bool(writer.pending.any()) and not bool(writer.attempt_counts.any())
