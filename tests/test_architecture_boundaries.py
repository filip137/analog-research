"""Ownership and loader invariants which must survive future extensions."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('package,forbidden', [
    ('core', {'hardware', 'networks', 'methods', 'ebl', 'experiments', 'campaigns', 'workflow', 'training'}),
    ('hardware', {'networks', 'methods', 'ebl', 'experiments', 'campaigns', 'workflow', 'training'}),
    ('networks', {'methods', 'ebl', 'experiments', 'campaigns', 'workflow', 'training'}),
    ('methods', {'ebl', 'experiments', 'campaigns', 'workflow', 'training'}),
])
def test_domain_imports_follow_ownership(package, forbidden):
    violations = []
    for path in sorted((ROOT / package).rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name.split('.')[0] in forbidden:
                    violations.append(f'{path.relative_to(ROOT)}:{node.lineno}: {name}')
    assert not violations, '\n'.join(violations)


def test_lifecycle_parser_stays_stdlib_only():
    subprocess.run([sys.executable, '-B', '-c',
                    "import sys; from ebl.cli import build_parser; build_parser(); "
                    "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                   cwd=ROOT, check=True, capture_output=True, text=True)


def test_restricted_checkpoint_failure_never_retries_unrestricted(tmp_path, monkeypatch):
    from methods import checkpoint
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'invalid')
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError('invalid checkpoint')
    monkeypatch.setattr(checkpoint.torch, 'load', fail)
    with pytest.raises(TypeError, match='invalid checkpoint'):
        checkpoint._torch_load(path)
    assert calls == [{'map_location': 'cpu', 'weights_only': True}]


def test_workflow_is_documentation_only():
    assert not list((ROOT / 'workflow').rglob('*.py'))


def test_physical_fault_groups_follow_supplied_domains():
    from hardware.arrays import hierarchical_faults
    from hardware.config import DefectCase
    from hardware.mapping import FaultDomain, MatrixSpec
    # A suffix which once implicitly meant "adapter" has no meaning to hardware.
    layout = [MatrixSpec('arbitrary.lora_a', (16, 16), 0, 256, 1.)]
    case = DefectCase('random', 50000, hierarchical=True, target='base')
    result = hierarchical_faults(layout, 8, 12, case, 271001, 'declared-namespace',
                                  domains={'arbitrary.lora_a': FaultDomain('declared-group', False)})
    assert result['receipt']['failed_devices'] == round(512 * .05)
    assert result['receipt']['groups'][0]['group'] == 'declared-group'
    with pytest.raises(ValueError, match='one fault-domain'):
        hierarchical_faults(layout, 8, 12, case, 271001, 'declared-namespace', domains={})


def test_array_constructor_rejects_unknown_physical_technology():
    from hardware.arrays import fresh_array
    with pytest.raises(ValueError, match='Unsupported physical technology'):
        fresh_array(technology='unregistered', layout=(), assignment_seed=1, case=None,
                    populations={}, device='cpu', namespace='test', endpoint_seed=2,
                    noise_scale=1, tile_size=8, arrays_per_die=1, acceptance_percent=2, domains={})
