"""Tiny mathematical fixtures. Task experiments remain GPU FP32 only."""
import math
import pytest
import torch

from networks.energy_transformer.gpu import check_domain, EquilibriumError, normalize
from networks.energy_transformer.input_attention import InputAttentionNetwork, project, norms, certificate
from networks.energy_transformer.finite_time import finite_ep, bptt_gradient, agreement, relax

torch.set_num_threads(1)


def case():
    m = InputAttentionNetwork(tokens=3, width=8, patch_width=2, heads=2, dtype=torch.float64)
    project(m)
    gen = torch.Generator().manual_seed(814)
    x = torch.randn(2, 3, 4, generator=gen, dtype=torch.float64)*.3
    target = torch.randn(2, 3, 2, generator=gen, dtype=torch.float64)*.2
    z = torch.randn(2, 3, 8, generator=gen, dtype=torch.float64)*.6
    y = torch.randn(2, 3, 2, generator=gen, dtype=torch.float64)*.3
    return m, x, target, torch.ones_like(target), z, y


@pytest.mark.parametrize('intervention', ['normal', 'uniform', 'zero'])
def test_force_and_every_local_statistic_are_derivatives_of_one_energy(intervention):
    m, x, target, mask, z, y = case()
    for p in m.parameters(): p.requires_grad_(True)
    z.requires_grad_(True); y.requires_grad_(True)
    ports = m.ports(z, y, intervention, context=m.prepare(x))
    dz, dy = m.force(ports, x, target, mask, -.005)
    energy = m.energy(ports, x, target, mask, -.005, intervention).sum()
    derivatives = torch.autograd.grad(energy, (z, y, *m.parameters()), allow_unused=True, retain_graph=True)
    expected_z = -torch.autograd.grad((ports['g']*dz.detach()).sum(), z)[0]
    torch.testing.assert_close(derivatives[0], expected_z, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(derivatives[1], -dy, rtol=2e-12, atol=2e-12)
    stats = m.statistics(ports, x)
    assert set(stats) == {n for n, _ in m.named_parameters()}
    for (name, p), actual in zip(m.named_parameters(), derivatives[2:]):
        torch.testing.assert_close(stats[name], torch.zeros_like(p) if actual is None else actual,
                                   rtol=2e-11, atol=2e-12)


def test_value_covariance_bounds_hessian_without_a_query_key_strength_penalty():
    m, x, _, _, _, _ = case()
    report = project(m)['after']
    g = torch.randn(3, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(912))
    g *= .8*math.sqrt(8)/g.norm(dim=-1, keepdim=True)
    context = m.prepare(x[:1])
    for sharpness in (1., 10., 100.):
        def interaction(flat):
            state = flat.reshape(3, 8)
            returned = torch.einsum('nd,hda->hna', state, m.output)
            logits = sharpness*context['scores'][0]+m.feedback*(returned @ context['value'][0].transpose(-1, -2))
            return m.gain/m.feedback*torch.logsumexp(logits, -1).sum()
        hessian = torch.autograd.functional.hessian(interaction, g.flatten())
        eigen = torch.linalg.eigvalsh(hessian)
        assert eigen.min() > -1e-12
        assert eigen.max() <= report['attention']+1e-12


def test_strong_attention_and_contraction_can_coexist():
    m = InputAttentionNetwork(dtype=torch.float64)
    direction = torch.zeros(64, dtype=torch.float64)
    direction[:32:2] = 1/math.sqrt(32)
    direction[1:32:2] = -1/math.sqrt(32)
    with torch.no_grad():
        m.query.zero_(); m.key.zero_()
        m.query[0, 0] = .5*direction; m.key[0, 0] = .5*direction
    report = project(m)['after']
    # Synthetic input outside the MNIST voltage range, passed through the
    # actual fixed-feature preprocessing. This is a capacity witness only.
    x = (-100*direction[:32]).expand(1,49,32).clone()
    x[:,0] *= -1
    context = m.prepare(x)
    z = torch.zeros(1, 49, 64, dtype=torch.float64)
    y = torch.zeros(1, 49, 16, dtype=torch.float64)
    ports = m.ports(z, y, context=context)
    assert ports['p'][0, 0, 0, 0].item() > .98
    assert context['f'].norm(dim=-1).max() <= 8
    assert report['normalized_total'] <= .7
    assert report['contraction_radius'] < 1
    assert check_domain(m, ports)['valid'].all()


def test_projection_preserves_legal_routing_when_memory_or_values_grow():
    m, x, _, _, _, _ = case()
    before = {n: getattr(m, n).clone() for n in ('query', 'key', 'encoder', 'position')}
    with torch.no_grad():
        m.memory.mul_(10); m.value.mul_(10); m.output.mul_(10)
    result = project(m)
    for n, p in before.items(): assert torch.equal(getattr(m, n), p)
    assert result['after']['normalized_total'] <= .7
    assert m.fixed_features(x).norm(dim=-1).max() <= math.sqrt(m.width)
    again = project(m)['after']
    for key, value in result['after'].items():
        assert again[key] == pytest.approx(value, rel=2e-12, abs=2e-12)


def test_joint_update_satisfies_the_weighted_contraction_bound():
    m, x, target, mask, z, y = case()
    x = x[:1].expand_as(x); target = target[:1].expand_as(target)
    n = norms(m); report = certificate(m, n)
    a = (report['attention']+report['memory'])/math.sqrt(m.epsilon)
    weights = (float(n['decoder']), report['contraction_radius']-a)
    def distance(z, y):
        return max(float((z[0]-z[1]).norm())/weights[0], float((y[0]-y[1]).norm())/weights[1])
    dz, dy = m.force(m.ports(z, y, context=m.prepare(x)), x, target, mask, -.005)
    assert distance(z+dz, y+dy/1.3) <= report['contraction_radius']*distance(z, y)+1e-12


def test_ep_includes_independent_value_output_and_matches_bptt_and_finite_differences():
    m, x, target, mask, _, _ = case()
    ep, phases = finite_ep(m, x, target, mask, 64)
    bp, phase = bptt_gradient(m, x, target, mask, 64)
    metrics = agreement(ep, bp)
    assert set(metrics['groups']) == {n for n, _ in m.named_parameters()}
    assert metrics['global']['cosine'] > .99999
    for group in metrics['groups'].values():
        assert group['cosine'] > .99999
        assert group['relative_l2'] < .0001
    assert all(p.residual < 1e-11 for p in phases)
    torch.testing.assert_close(phases[0].y, phase.y, atol=0, rtol=0)
    gen = torch.Generator().manual_seed(115)
    for name in ('query', 'key', 'value', 'output'):
        p = getattr(m, name); original = p.clone()
        direction = torch.randn(p.shape, dtype=p.dtype, generator=gen)
        direction /= direction.norm(); costs = []
        for sign in (-1, 1):
            p.copy_(original+sign*1e-4*direction)
            result = relax(m, x, target, mask, 64)
            costs.append(float(.5*(mask*(result.y-target).square()).sum()/mask.sum()))
        p.copy_(original)
        numerical = (costs[1]-costs[0])/2e-4
        assert float((ep[name]*direction).sum()) == pytest.approx(numerical, rel=.002, abs=1e-10)


@pytest.mark.parametrize('operand', ['value', 'returned_state'])
def test_domain_checks_new_feedback_and_value_operands(operand):
    m, x, _, _, z, y = case()
    ports = m.ports(z, y, context=m.prepare(x))
    ports[operand] = torch.full_like(ports[operand], 6.1)
    with pytest.raises(EquilibriumError, match='drain_margin'):
        check_domain(m, ports)


def test_gradient_agreement_cannot_silently_omit_a_new_parameter_group():
    a = {'query': torch.ones(2), 'value': torch.ones(2)}
    with pytest.raises(ValueError, match='groups differ'):
        agreement(a, {'query': a['query']})


def test_public_recipes_build_identical_ep_bp_models_and_keep_all_groups_trainable():
    import json
    from pathlib import Path
    from ebl.schema.energy_transformer_mnist import parse_config
    from ebl.runtime.energy_transformer_mnist import build_model, initialize_model, core_parameter_count
    root = Path(__file__).resolve().parents[1]/'examples/energy_transformer_mnist/input_attention'
    configs = [parse_config(json.loads((root/f'input-{arm}.json').read_text())) for arm in ('energy_ep','energy_bp')]
    models = [build_model(c, device='cpu') for c in configs]
    for m,c in zip(models,configs): initialize_model(m,c)
    assert set(models[0].state_dict()) == set(models[1].state_dict())
    for name,value in models[0].state_dict().items(): assert torch.equal(value,models[1].state_dict()[name])
    assert core_parameter_count(models[0]) == 8*64*64
    assert sum(p.numel() for p in models[0].parameters()) == 38976
    recurrent = parse_config(json.loads((root/'recurrent-energy_bp.json').read_text()))
    control = build_model(recurrent, device='cpu')
    initialize_model(control,recurrent)
    assert core_parameter_count(control) == 6*64*64


def test_invalid_architecture_and_policy_combinations_are_rejected():
    from ebl.schema.energy_transformer_mnist import parse_config
    from ebl.schema.base import ConfigError
    for config in ({'energy_architecture':'anything'}, {'arm':'energy_bp'},
                   {'energy_architecture':'input_conditioned'},
                   {'recurrent_constraint':'input_attention_v1'},
                   {'initialization':'input_attention_v1'},
                   {'arm':'digital_bp','energy_architecture':'input_conditioned'}):
        with pytest.raises(ConfigError): parse_config(config)
