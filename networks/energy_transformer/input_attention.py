"""Input-conditioned variational attention with independent Q/K/V/output maps.

This is an experimental energy block, not a transistor-level implementation.
Input features are fixed functions of the observed input and token index. All
eight learned maps use local EP statistics; no learned feature extractor is
silently differentiated outside EP. Returned values feed back into the softmax
scores, as required by the scalar energy. Setting that feedback to zero is NOT
an EP-compatible shortcut for training query/key weights.
"""
import math
import torch
from torch import nn

from networks.energy_transformer.gpu import (
    EnergyNetwork, EquilibriumError, diode, initialize, normalize, mos_domain,
)
from networks.energy_transformer.contractive import BUDGETS


class InputAttentionNetwork(EnergyNetwork):
    def __init__(self, tokens=49, width=64, patch_width=16, heads=4,
                 seed=20261114, epsilon=.5, device='cpu', dtype=torch.float32,
                 device_domain='physical', solver_dt=1.,
                 solver_rule='energy_backtracking', feedback=.1, gain=1., score_gain=4.):
        if width % 4 or width % heads:
            raise ValueError('Width must be divisible by four and by heads')
        if device_domain != 'physical':
            raise ValueError('Input attention requires all physical domain checks')
        if not math.isfinite(feedback) or feedback <= 0:
            raise ValueError('Feedback must be strictly positive for EP')
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError('Attention gain must be positive')
        if not math.isfinite(score_gain) or score_gain <= 0:
            raise ValueError('Input score gain must be positive')
        self.score_gain = score_gain
        super().__init__(tokens, width, patch_width, heads, seed, epsilon,
                         device, dtype, device_domain, solver_dt, solver_rule)
        self.feedback, self.gain = feedback, gain
        self.feature_width = 2*patch_width + width//2
        dim = width//heads
        for name in ('query', 'key', 'value'):
            setattr(self, name, nn.Parameter(torch.empty(
                heads, dim, self.feature_width, device=device, dtype=dtype),
                requires_grad=False))
        self.output = nn.Parameter(torch.empty(heads, width, dim,
                                  device=device, dtype=dtype), requires_grad=False)
        index = torch.arange(tokens, device=device, dtype=dtype)[:, None]
        frequency = torch.exp(-math.log(10000.)*torch.arange(
            width//4, device=device, dtype=dtype)/(width//4))
        angle = index*frequency
        self.register_buffer('fixed_position', torch.cat((angle.sin(), angle.cos()), -1))
        initialize(self, seed)

    def fixed_features(self, x):
        """Bounded, nonlearned preprocessing; no parameter-gradient bypass."""
        f = torch.cat((x, self.fixed_position.expand(len(x), -1, -1)), -1)
        return normalize(f, self.epsilon)[0]*math.sqrt(self.width/self.feature_width)

    def attention_context(self, f):
        """Algebraic input ports. The public prepare path bounds ||f_i||<=sqrt(D)."""
        q = torch.einsum('bnf,haf->bhna', f, self.query)
        k = torch.einsum('bnf,haf->bhna', f, self.key)
        value = torch.einsum('bnf,haf->bhna', f, self.value)
        scores = self.score_gain*self.kappa*(q @ k.transpose(-1, -2))
        return dict(f=f, q=q, k=k, value=value, scores=scores,
                    base_probability=scores.softmax(-1))

    def prepare(self, x):
        return self.attention_context(self.fixed_features(x))

    def ports(self, z, y, attention='normal', *, context=None):
        if context is None:
            raise ValueError('Input attention needs prepared input ports')
        g, s = normalize(z, self.epsilon)
        returned_state = torch.einsum('bnd,hda->bhna', g, self.output)
        logits = context['scores'] + self.feedback*(
            returned_state @ context['value'].transpose(-1, -2))
        if attention == 'normal':
            p = logits.softmax(-1)
            # The energy subtracts a state-independent input potential. This
            # centering reduces cancellation in query/key phase statistics.
            delta_p = p-context['base_probability']
        elif attention == 'uniform':
            p = torch.full_like(logits, 1/self.tokens)
            delta_p = torch.zeros_like(p)
        elif attention == 'zero':
            p = torch.zeros_like(logits)
            delta_p = torch.zeros_like(p)
        else:
            raise ValueError('Unknown attention intervention')
        return dict(context, z=z, y=y, g=g, s=s, returned_state=returned_state,
                    logits=logits, p=p, delta_p=delta_p,
                    r=diode(g @ self.memory.T))

    def force(self, v, x, target, mask, beta=0.):
        returned_value = v['p'] @ v['value']
        drive = x @ self.encoder.T + self.position
        drive = drive + self.gain*torch.einsum('bhna,hda->bnd', returned_value, self.output)
        dz = drive + v['r'] @ self.memory + v['y'] @ self.decoder - v['z']
        dy = v['g'] @ self.decoder.T - v['y'] - beta*mask*(v['y']-target)
        return dz, dy

    def energy(self, v, x, target, mask, beta=0., attention='normal'):
        total = (v['z']*v['g']).sum((1, 2)) - self.width*v['s'].sum((1, 2))
        total -= ((x @ self.encoder.T + self.position)*v['g']).sum((1, 2))
        if attention == 'normal':
            # Subtracted baseline depends on weights, but not on dynamical
            # states. Include its derivative in statistics as delta_p.
            difference = torch.logsumexp(v['logits'], -1)-torch.logsumexp(v['scores'], -1)
            total -= self.gain/self.feedback*difference.sum((1, 2))
        elif attention == 'uniform':
            total -= self.gain*(v['returned_state']*v['value'].mean(-2, keepdim=True)).sum((1, 2, 3))
        elif attention != 'zero':
            raise ValueError('Unknown attention intervention')
        total -= .5*v['r'].square().sum((1, 2))
        total += (.5*v['y'].square()-v['y']*(v['g'] @ self.decoder.T)).sum((1, 2))
        return total + .5*beta*(mask*(v['y']-target).square()).sum((1, 2))

    def statistics(self, v, x):
        factor = self.gain*self.score_gain*self.kappa/self.feedback
        return {
            'query': -factor*torch.einsum('bhna,bnf->haf', v['delta_p'] @ v['k'], v['f']),
            'key': -factor*torch.einsum('bhna,bnf->haf', v['delta_p'].transpose(-1, -2) @ v['q'], v['f']),
            'value': -self.gain*torch.einsum('bhna,bnf->haf', v['p'].transpose(-1, -2) @ v['returned_state'], v['f']),
            'output': -self.gain*torch.einsum('bnd,bhna->hda', v['g'], v['p'] @ v['value']),
            'memory': -torch.einsum('bnm,bnd->md', v['r'], v['g']),
            'encoder': -torch.einsum('bnd,bnp->dp', v['g'], x),
            'decoder': -torch.einsum('bnp,bnd->pd', v['y'], v['g']),
            'position': -v['g'].sum(0),
        }

    @torch.no_grad()
    def device_margins(self, v):
        """Forward QK, output-feedback and probability-value MOS products.

        Linear projection sums have ideal range here. Weight programming and
        the phase-statistic readout are not transistor-validated by this check.
        """
        d = mos_domain(v)  # Includes the weak-inversion softmax bank.
        operands = {key: v[key].abs() for key in ('q', 'k', 'value', 'returned_state')}
        maxima = torch.stack([a.amax((1, 2, 3)) for a in operands.values()])
        maximum = maxima.amax(0)
        score = (operands['q'].amax(-2)+operands['k'].amax(-2)).amax((1, 2))
        feedback = (operands['returned_state'].amax(-2)+operands['value'].amax(-2)).amax((1, 2))
        readout = v['p'].abs().amax((1, 2, 3))+operands['value'].amax((1, 2, 3))
        triode = .7-.05*torch.stack((score, feedback, readout)).amax(0)
        drain = .3-.05*maximum
        weak = d['weak_inversion_margin']
        valid = (torch.isfinite(maximum) & torch.isfinite(weak) &
                 (triode > 0) & (drain > 0) & (weak > 0))
        return dict(maximum_operand=maximum, triode_margin=triode,
                    drain_margin=drain, weak_inversion_margin=weak, valid=valid)


@torch.no_grad()
def norms(model):
    return {name: torch.linalg.matrix_norm(getattr(model, name), ord=2)
            for name in ('query', 'key', 'value', 'output', 'memory', 'decoder')}


def certificate(model, n, beta=.005):
    # ||u_hj|| <= sqrt(D) ||O_h|| ||V_h||. The attention Jacobian is
    # gain*feedback*Cov_p(u), bounded by gain*feedback*max_j||u_j||^2.
    radius = math.sqrt(model.width)*n['output']*n['value']
    attention = float(model.gain*model.feedback*radius.square().sum())
    memory = float(n['memory'].square())
    decoder_norm = float(n['decoder'])
    decoder = decoder_norm**2/(1-beta)
    a = (attention+memory)/math.sqrt(model.epsilon)
    c = 1-(1-beta)/1.3
    rho = .5*(a+c+math.sqrt((a-c)**2+4*decoder_norm**2/(1.3*math.sqrt(model.epsilon))))
    return dict(attention=attention, memory=memory, decoder=decoder,
                total=attention+memory+decoder,
                normalized_total=(attention+memory+decoder)/math.sqrt(model.epsilon),
                contraction_radius=rho, value_radius=radius.tolist())


@torch.no_grad()
def project(model, beta=.005):
    """Separate block budgets; Q/K have only an operand-range norm cap.

    This remains a digital programming operation. Spectral constraints are not
    claimed to be a local analog learning rule.
    """
    if not 0 < beta < 1:
        raise ValueError('Expected 0 < beta < 1')
    before = certificate(model, norms(model), beta)
    threshold = math.sqrt(model.epsilon)*(1-1e-4)
    # Every projected multiplier operand is <=4 for bounded f and g.
    cap = 4/math.sqrt(model.width)
    for name in ('query', 'key', 'value', 'output'):
        p = getattr(model, name)
        scale = (cap/torch.linalg.matrix_norm(p, ord=2).clamp_min(1e-30)).clamp_max(1.)
        p.mul_(scale[:, None, None])
    n = norms(model)
    current = certificate(model, n, beta)
    scale = min(1., (BUDGETS['attention']*threshold/max(current['attention'], 1e-30))**.25)
    model.value.mul_(scale)
    model.output.mul_(scale)
    model.memory.mul_(min(1., math.sqrt(BUDGETS['memory']*threshold)/max(float(n['memory']), 1e-30)))
    model.decoder.mul_(min(1., math.sqrt(BUDGETS['decoder']*threshold*(1-beta))/max(float(n['decoder']), 1e-30)))
    after = certificate(model, norms(model), beta)
    if not math.isfinite(after['total']) or after['normalized_total'] > .7:
        raise EquilibriumError('Input-attention projection failed its certificate')
    for name, portion in BUDGETS.items():
        if after[name] > portion*math.sqrt(model.epsilon):
            raise EquilibriumError('Input-attention block budget exceeded')
    return dict(before=before, after=after, budgets=dict(BUDGETS), operand_norm_cap=cap)
