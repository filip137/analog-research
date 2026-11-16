"""Explicit MOS bump-chain attention scorer (on-chip-learning #55, Phase 2).

Per key j and score dimension d: an NMOS differential pair (gates q_d, k_jd) with
diode-connected PMOS loads carrying I1 and I2, and a series PMOS current correlator
whose current is I1*I2/(I1+I2) = I_tail * sech^2(kappa*(q_d-k_jd)/(2*U_T)) / 4.
An NMOS mirror passes the dimension-0 correlator current to the dimension-1 tail;
the dimension-1 correlator drains directly into the key's attention controller
node g_a<j> of the pseudoconductance bank. First-stage tails copy one reference.

Query/key voltages and the tail reference current are ideal external ports.
A zero-volt probe Vsc_<j> measures each delivered attention current.
"""
import numpy as np

UT = 8.617333262145e-5 * 300.15
DIMS = 2
KEYS = 3


def law(query, keys, tail_current, iref, kappa):
    """Ideal attention currents a_j (units of iref)."""
    d = np.asarray(query)[None, :] - np.asarray(keys)
    return tail_current / iref * np.prod(0.25 / np.cosh(kappa * d / (2 * UT)) ** 2, axis=1)


def law_gradients(query, keys, tail_current, iref, kappa):
    """d a_j / d q_d (3x2) and d a_j / d k_jd (3x2) of the ideal law."""
    a = law(query, keys, tail_current, iref, kappa)
    d = np.asarray(query)[None, :] - np.asarray(keys)
    slope = -(kappa / UT) * np.tanh(kappa * d / (2 * UT))     # d ln a / d q_d
    dq = a[:, None] * slope
    return dq, -dq


def add_scorer(lines, mos, scorer):
    """Append scorer devices/sources to a pseudoconductance netlist; return columns."""
    query = np.asarray(scorer['query'], dtype=float)
    keys = np.asarray(scorer['keys'], dtype=float)
    tail = float(scorer['tail_current'])
    if query.shape != (DIMS,) or keys.shape != (KEYS, DIMS):
        raise ValueError('Expected a 2-D query and three 2-D keys')
    if not 0 < tail <= 2e-9:
        raise ValueError('Tail current must keep the scorer in weak inversion (<=2 nA)')
    for d in range(DIMS):
        lines.append(f'Vq{d} q{d} 0 {query[d]:.17g}')
    for j in range(KEYS):
        for d in range(DIMS):
            lines.append(f'Vk{j}{d} k{j}{d} 0 {keys[j, d]:.17g}')
    lines.append(f'Iref_tail vdd tref {tail:.17g}')
    mos('tref', 'tref', 'tref', '0', '0', 'nc', 'scorer')
    lines.append('.nodeset v(tref)=.6')
    columns = []
    for j in range(KEYS):
        mos(f'tail_{j}', f't{j}0', 'tref', '0', '0', 'nc', 'scorer')
        for d in range(DIMS):
            tail_node, a, b, c = f't{j}{d}', f'pa{j}{d}', f'pb{j}{d}', f'c{j}{d}'
            out = f'o{j}{d}' if d < DIMS - 1 else f'so{j}'
            mos(f'dpa_{j}{d}', a, f'q{d}', tail_node, '0', 'nc', 'scorer')
            mos(f'dpb_{j}{d}', b, f'k{j}{d}', tail_node, '0', 'nc', 'scorer')
            mos(f'lda_{j}{d}', a, a, 'vdd', 'vdd', 'pc', 'scorer')
            mos(f'ldb_{j}{d}', b, b, 'vdd', 'vdd', 'pc', 'scorer')
            mos(f'coa_{j}{d}', c, a, 'vdd', 'vdd', 'pc', 'scorer')
            mos(f'cob_{j}{d}', out, b, c, 'vdd', 'pc', 'scorer')
            lines.append(f'.nodeset v({a})=1.05 v({b})=1.05 v({c})=1.7 v({tail_node})=.45')
            if d < DIMS - 1:
                mos(f'md_{j}{d}', out, out, '0', '0', 'nc', 'scorer')
                mos(f'mc_{j}{d}', f't{j}{d + 1}', out, '0', '0', 'nc', 'scorer')
                lines.append(f'.nodeset v({out})=.55')
        lines.append(f'Vsc_{j} so{j} g_a{j} 0')
        columns.append((f'score_{j}', f'i(Vsc_{j})'))
        columns += [(f'tail_{j}{d}', f'v(t{j}{d})') for d in range(DIMS)]
    return columns


def measured(port, iref):
    return np.array([port[f'score_{j}'][-1] / iref for j in range(KEYS)])


def min_tail_voltage(port):
    return min(float(port[f'tail_{j}{d}'][-1]) for j in range(KEYS) for d in range(DIMS))
