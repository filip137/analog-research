import copy
import torch
import pytest
from experiments.cifar_crossbar.devices import PulseWriter
from experiments.cifar_crossbar.closed_loop_lr_updates import UncappedClosedLoopAdam
from experiments.cifar_crossbar.closed_loop_lr_config import default_config, parse_config
from experiments.cifar_crossbar.closed_loop_lr_runtime import selected_rates


class Port:
    nominal_step = .0949
    def __init__(self, size=17):
        self.size = size; self.state = torch.zeros(size); self.count = torch.zeros(size, dtype=torch.int64)
    def read(self): return self.state.clone()
    def pulse(self, direction):
        self.state.add_(direction * self.nominal_step).clamp_(-1, 1); self.count += direction.abs().to(torch.int64)
    def finish_step(self): pass


def test_closed_loop_matches_legacy_exactly_before_cap():
    a, b = Port(), Port()
    modern = UncappedClosedLoopAdam(a, a.state, learning_rate=.001, seed=7)
    legacy = PulseWriter(b, method='closed_loop', learning_rate=.001, tolerance=.04745, cap=640, maximum_pulses=1, seed=7)
    rng = torch.Generator().manual_seed(11)
    for _ in range(1000):
        grad = torch.randn(17, generator=rng) + .4
        modern.step(grad); legacy.step(grad)
        for field in ('target', 'first', 'second', 'count'):
            assert torch.equal(getattr(modern, field), getattr(legacy, field))
        assert torch.equal(a.state, b.state)
    assert modern.verify_reads == legacy.verify_reads


def test_no_total_cap_but_one_pulse_per_update_and_exact_restart():
    port = Port(1); writer = UncappedClosedLoopAdam(port, port.state, learning_rate=.1)
    for _ in range(700):
        before = writer.count.clone(); writer.step(torch.ones(1))
        assert bool((writer.count - before <= 1).all())
    assert int(writer.count[0]) == 700
    state = writer.state_dict(); physical = port.state.clone()
    other = Port(1); other.state.copy_(physical)
    restored = UncappedClosedLoopAdam(other, other.state, learning_rate=.1)
    restored.load_state_dict(state)
    writer.step(torch.tensor([-.3])); restored.step(torch.tensor([-.3]))
    for field in ('target', 'first', 'second', 'count'):
        assert torch.equal(getattr(writer, field), getattr(restored, field))
    assert writer.state_dict()['contract'][3:] == (None, 1)


def test_frozen_lr_grid_partition_and_controls():
    for rate in (.0003, .001):
        with pytest.raises(ValueError): default_config(learning_rates=[rate])
    for change in ({'verify_step_fraction': .1}, {'recovery_pulses': 640}, {'update_pulses': 2},
                   {'array_seed': 251001, 'endpoint_seed': 261001}, {'epochs': 1}, {'calibration_lr': .001}):
        with pytest.raises(ValueError): default_config(**change)
    default_config(phase='confirmation', array_seed=251001, endpoint_seed=261001)
    default_config(max_examples=128, array_seed=271001, endpoint_seed=281001)


def test_confirmation_requires_explicit_development_selection_and_deduplicates_baseline():
    spec = parse_config(default_config(phase='confirmation', array_seed=251001, endpoint_seed=261001))
    with pytest.raises(ValueError): selected_rates(spec, None, 'hash')
    receipt = {'schema': 'cifar_om_closed_loop_lr.selection.v1', 'development_arrays': [211001, 211002],
               'criterion': 'mean_epoch5_development_teacher_kl',
               'groups': {'cifar10_4': {'source_checkpoint_sha256': 'hash', 'selected_learning_rate': .001}}}
    assert selected_rates(spec, receipt, 'hash') == (.0001, .001)
    receipt['groups']['cifar10_4']['selected_learning_rate'] = .0001
    assert selected_rates(spec, receipt, 'hash') == (.0001,)
    receipt['development_arrays'] = [251001, 251002]
    with pytest.raises(ValueError): selected_rates(spec, receipt, 'hash')


def test_selection_uses_all_development_cells_and_rejects_test_leakage():
    from experiments.cifar_crossbar.closed_loop_lr_config import LRS, SOURCES, CASES, DEV_SEEDS
    from experiments.cifar_crossbar.closed_loop_lr_report import choose
    from experiments.cifar_crossbar.hwa_fault_runtime import case_name
    results = []
    for dataset in ('cifar10', 'cifar100'):
        for suffix in ('last_two_blocks', 'last_four_blocks'):
            for seed in DEV_SEEDS:
                rows = []
                for source in SOURCES:
                    for case in CASES:
                        for rate in (None, *LRS):
                            score = {None: 3., .0001: 2., .0003: 1., .001: 1.}[rate]
                            rows.append({'source': source, 'case': case_name(*case),
                                'method': 'calibration' if rate is None else 'onchip_calibration',
                                'learning_rate': rate, 'p0_sha256': str((source, case, seed)), 'initial': {},
                                'curve': [{'epoch': epoch, 'image_presentations': 45000 * epoch,
                                    'development': {'teacher_kl': score},
                                    'cost': {'max_recovery_cell_pulses': 0}} for epoch in range(1, 6)]})
                results.append({'dataset': dataset, 'suffix': suffix, 'phase': 'development', 'array_seed': seed,
                    'source_checkpoint_sha256': dataset + suffix, 'smoke_only': False, 'recovery_pulse_cap': None,
                    'per_update_pulse_cap': 1, 'verify_tolerance': .04745, 'epochs': 5, 'test_evaluated': False,
                    'learning_rates': LRS, 'exact_replays': 24, 'completed_controls': 24, 'measurements': rows})
    receipt = choose(results)
    assert all(value['selected_learning_rate'] == .0003 for value in receipt['groups'].values())
    assert all(value['endpoints_per_rate'] == 12 for value in receipt['groups'].values())
    with pytest.raises(ValueError): choose(results[:-1])
    leaked = copy.deepcopy(results)
    leaked[0]['measurements'][0]['curve'][0]['test'] = {'teacher_kl': 0.}
    with pytest.raises(ValueError): choose(leaked)
