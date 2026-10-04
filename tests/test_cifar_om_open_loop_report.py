from copy import deepcopy
import pytest
from experiments.cifar_crossbar.om_open_loop_config import OmOpenLoopSpec
from experiments.cifar_crossbar.om_open_loop_report import validate
from experiments.cifar_crossbar.sweep_config import SOURCES, cases
from experiments.cifar_crossbar.hwa_fault_runtime import case_name


def fixture():
    spec = OmOpenLoopSpec(); rows = []
    metric = {'examples': 10000, 'accuracy_percent': 80., 'teacher_kl': .5, 'teacher_agreement_percent': 85.}
    for source in SOURCES:
        for kind, rate in cases(spec, source):
            label = case_name(kind, rate)
            for method in spec.methods:
                curve = []
                if method != 'none':
                    for epoch in range(1, 6):
                        count = 0 if method == 'calibration' else 704 * epoch
                        curve.append({'epoch': epoch, 'test': metric, 'image_presentations': 45000 * epoch,
                                      'cost': {'verify_reads': 0, 'capped_cells': 0, 'physical_pulses': count,
                                               'max_recovery_cell_pulses': count}})
                rows.append({'source': source, 'case': label, 'method': method, 'identity': label,
                             'p0_sha256': source + label, 'initial_apparent_sha256': source + label,
                             'initial_test': metric, 'programming': {}, 'curve': curve})
    return {'measurements': rows, 'completed_controls': 216, 'exact_replays': 162,
            'epochs': 5, 'recovery_images': 45000, 'smoke_only': False,
            'per_update_pulse_cap': 1, 'recovery_pulse_cap': None, 'verification_reads': 0}


def test_full_coverage_accepts_counts_above_legacy_cap():
    assert len(validate(fixture())) == 216


@pytest.mark.parametrize('change', ['missing', 'cap', 'reads', 'pulse_count', 'calibration', 'pairing', 'replay'])
def test_invalid_open_loop_evidence_is_rejected(change):
    result = deepcopy(fixture())
    if change == 'missing': result['measurements'].pop()
    elif change == 'cap': result['recovery_pulse_cap'] = 640
    elif change == 'reads': result['measurements'][2]['curve'][0]['cost']['verify_reads'] = 1
    elif change == 'pulse_count': result['measurements'][2]['curve'][0]['cost']['max_recovery_cell_pulses'] = 705
    elif change == 'calibration': result['measurements'][1]['curve'][0]['cost']['physical_pulses'] = 1
    elif change == 'pairing': result['measurements'][1]['p0_sha256'] = 'different'
    else: result['exact_replays'] = 161
    with pytest.raises(ValueError): validate(result)
