import pytest
import torch
from methods import fault_tuning as ft
from methods.opt_recovery import RecoveryWritePort
from methods.cmo_direct_adam import ThresholdedBoundedWriteAdam
from networks.opt.physical import CmoDrnBank
from tests.test_opt_cmo_drn import cpu_threads
from tests.test_opt_cmo_drn_fault_tuning import bank_and_model


def test_calibration_identity_updates_only_digital_correction_and_roundtrips(cpu_threads):
    model, _, state = bank_and_model()
    bank = CmoDrnBank(state, 'cpu')
    tokens = torch.randint(0, 17, (2, 6))
    samples = torch.randint(0, 17, (2, 5, 8))
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    held = bank.held_hashes()
    with torch.no_grad(): before = model(tokens, bank.view(model))
    arm = ft.CalibrationArm(model, bank, .01, 3)
    with torch.no_grad(): torch.testing.assert_close(model(tokens, bank.view(model)), before, rtol=0, atol=0)
    for _ in range(3): arm.step(tokens, samples, 1, 'cpu')
    with torch.no_grad(): after = model(tokens, bank.view(model))
    assert not torch.equal(after, before)
    saved = arm.state()
    assert bank.held_hashes() == held and bank.costs()['write_attempts'] == 0
    for k, v in model.state_dict().items(): torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
    arm.close()
    with torch.no_grad(): torch.testing.assert_close(model(tokens, bank.view(model)), before, rtol=0, atol=0)
    replay = ft.CalibrationArm(model, bank, .01, 3)
    replay.adapters.load_state_dict(saved)
    with torch.no_grad(): torch.testing.assert_close(model(tokens, bank.view(model)), after, rtol=0, atol=0)
    replay.close()


def test_matched_noiseless_gate_retains_nominal_noise_and_thresholded_writer(cpu_threads):
    model, _, state = bank_and_model()
    bank = CmoDrnBank(state, 'cpu')
    arm = ft.PhysicalArm(model, bank, 'matched_noiseless', .001, 3)
    assert all(isinstance(w, ThresholdedBoundedWriteAdam) for w in arm.writers)
    for port in arm.ports:
        command = torch.full_like(port.array.held, .01)
        expected = RecoveryWritePort(port.array, 1., target_bounds='gmin_gmax').write_noise_std_q(command)
        torch.testing.assert_close(port.write_noise_std_q(command), expected, rtol=0, atol=0)
        assert bool((expected > 0).all()) and port.noise_factor == 0
    # The same command issues the same gate decisions from an identical held state.
    noisy = ft.PhysicalArm(model, CmoDrnBank(state, 'cpu'), 'noisy', .001, 3)
    for exact, stochastic in zip(arm.writers, noisy.writers):
        gradient = torch.ones_like(exact.port.array.held)
        exact.step(gradient); stochastic.step(gradient)
        torch.testing.assert_close(exact.attempt_counts, stochastic.attempt_counts, rtol=0, atol=0)
    arm.close(); noisy.close()


def test_unit_verifier_rejects_partial_duplicate_and_nonfinite(tmp_path):
    import importlib.util,json
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'campaigns/pilots/20261115-fault-tuning-completion/verify_unit.py'
    spec=importlib.util.spec_from_file_location('verify_fault_unit',script); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    config=tmp_path/'config.json'; run=tmp_path/'run';(run/'artifacts').mkdir(parents=True)
    config.write_text(json.dumps({'stage':'fault_tuning','fault_tuning':{'steps':300,'arms':[{'label':'cal','method':'calibration'}]}}))
    (run/'result.json').write_text(json.dumps({'status':'complete'}))
    row={'label':'cal','development':{'teacher_kl':.2},'updates':300}
    summary=run/'artifacts/summary.json'
    summary.write_text(json.dumps({'arms':[row]}));assert m.verify(config,run)['status']=='complete'
    for rows in [[],[row,row],[dict(row,updates=299)],[dict(row,development={'teacher_kl':float('nan')})]]:
        summary.write_text(json.dumps({'arms':rows}))
        with pytest.raises(ValueError):m.verify(config,run)
