import torch
from methods import fault_tuning as ft
from methods.onchip import ThresholdedWriteAdam
from networks.opt import OptMlpSuffix
from tests.test_workflow_opt import tiny_teacher


def test_sample_calibration_identity_frozen_base_and_replay():
    torch.set_num_threads(1)
    net = OptMlpSuffix(tiny_teacher(), 2, 8).eval()
    net.requires_grad_(False)
    tokens = torch.randint(0, 64, (2, 6)); samples = torch.randint(0, 64, (2, 5, 8))
    frozen = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad(): before = net(tokens, net.q)
    net.fault_output_calibration = ft.OutputCalibration(net)
    with torch.no_grad(): torch.testing.assert_close(net(tokens, net.q), before, rtol=0, atol=0)
    opt = torch.optim.Adam(net.fault_output_calibration.parameters(), lr=.01)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        ft.sample_loss(net(tokens, net.q), samples).backward(); opt.step()
    with torch.no_grad(): after = net(tokens, net.q)
    assert not torch.equal(after, before)
    saved = {k:v.clone() for k,v in net.fault_output_calibration.state_dict().items()}
    for k,v in frozen.items(): torch.testing.assert_close(net.state_dict()[k], v, rtol=0, atol=0)
    net.fault_output_calibration = None
    with torch.no_grad(): torch.testing.assert_close(net(tokens, net.q), before, rtol=0, atol=0)
    net.fault_output_calibration = ft.OutputCalibration(net)
    net.fault_output_calibration.load_state_dict(saved)
    with torch.no_grad(): torch.testing.assert_close(net(tokens, net.q), after, rtol=0, atol=0)


def test_noiseless_writes_keep_nominal_gate():
    class Array:
        def __init__(self): self.q=torch.zeros(4); self.commands=[]
        def read(self): return self.q.clone()
        def write(self, command, **kw): self.commands.append((command.clone(),kw))
    arrays=[Array(),Array()]
    writers=[ThresholdedWriteAdam(a,learning_rate=.1,threshold_sigma=2.,accumulate_updates=True,
        acceptance_percent=2.,noise_scale=1.,write_noise_scale=f) for a,f in zip(arrays,(1.,0.))]
    assert writers[0].threshold_q == writers[1].threshold_q > 0
    for w in writers: w.step(torch.ones(4))
    torch.testing.assert_close(arrays[0].commands[0][0],arrays[1].commands[0][0],rtol=0,atol=0)
    assert arrays[0].commands[0][1]['noise_factor']==1.
    assert arrays[1].commands[0][1]['noise_factor']==0.


def test_native_arm_loop_calibrates_and_cleans_up_without_writes():
    from types import SimpleNamespace as NS
    from ebl.runtime.fault_tuning import run_arm
    net=OptMlpSuffix(tiny_teacher(),2,8).eval(); net.requires_grad_(False)
    held=net.q.detach().clone()
    class Array:
        def read(self):return held.clone()
        def cost(self):return {}
        def write_counts(self):return torch.zeros_like(held,dtype=torch.int64)
        def verify_permanence(self):pass
        def out_of_range(self):return 0
    tokens=torch.randint(0,64,(4,6));samples=torch.randint(0,64,(4,5,8))
    cfg=NS(schedule={'steps':3,'development_every_steps':1,'order_seed':20261104,'batch':2},write_law={},lora_seed=271001)
    ctx=NS(network=net,cfg=cfg,device=torch.device('cpu'),array=lambda name:Array(),
           recovery=tokens,samples=samples,deployed_cost={},
           lc=NS(program_verify=NS(acceptance_percent=2.),devices=NS(noise_scale=1.)),
           stage=NS(store=NS(append_metric=lambda row:None)))
    def evaluate(q):
        with torch.no_grad(): return {'teacher_kl':float(ft.sample_loss(net(tokens,q),samples))}
    ctx.evaluate=evaluate
    before=evaluate(held)
    row,state=run_arm(ctx,ft.Arm('cal','calibration',.01))
    assert row['final_apparent_sha256']==row['initial_apparent_sha256']
    assert row['cost']['healthy_device_writes']==0 and state['schema'].endswith('calibration.v1')
    assert row['final']!=before and row['calibration_parameters']==64
    assert net.fault_output_calibration is None and evaluate(held)==before
