"""Passive differential resistive crossbar producing bump-scorer query/key voltages
(on-chip-learning #56, Phase B).

Each token x (3 inputs) is applied as V_cm +/- s*x_i by ideal sources. Each output
row d connects through G+_di to V_cm + s*x_i and G-_di to V_cm - s*x_i; the row is
unloaded (it drives only MOS gates), so ideally
    q_d = V_cm + s * sum_i (G+_di - G-_di) x_i / sum_i (G+_di + G-_di).
One crossbar copy per token: the query drives nodes q<d>, key j drives k<j><d>.
Ideal resistors: no programming error, drift, noise, mismatch or wire resistance.
"""
import numpy as np

INPUTS = 3
ROWS = 2


def formula(x, gp, gm, v_cm, scale):
    gp, gm = np.asarray(gp), np.asarray(gm)
    w = (gp - gm) / (gp + gm).sum(-1, keepdims=True)
    return v_cm + scale * np.asarray(x) @ w.T


def add_crossbar(lines, spec):
    """spec: tokens {'q': x(3,), 'k': x(3,3)}, gp, gm (2x3 siemens), v_cm, scale."""
    gp, gm = np.asarray(spec['gp'], float), np.asarray(spec['gm'], float)
    if gp.shape != (ROWS, INPUTS) or gm.shape != (ROWS, INPUTS) or min(gp.min(), gm.min()) <= 0:
        raise ValueError('Expected positive 2x3 conductance matrices')
    v_cm, scale = float(spec['v_cm']), float(spec['scale'])
    tokens = [('q', 'q{d}', np.asarray(spec['q'], float))]
    tokens += [(f'k{j}', f'k{j}{{d}}', np.asarray(spec['k'], float)[j]) for j in range(3)]
    columns = []
    for tag, row, x in tokens:
        if x.shape != (INPUTS,):
            raise ValueError('Expected three token inputs')
        for i in range(INPUTS):
            lines.append(f'Vx{tag}p{i} x{tag}p{i} 0 {v_cm + scale * x[i]:.17g}')
            lines.append(f'Vx{tag}m{i} x{tag}m{i} 0 {v_cm - scale * x[i]:.17g}')
        for d in range(ROWS):
            node = row.format(d=d)
            for i in range(INPUTS):
                lines.append(f'Rx{tag}p{d}{i} x{tag}p{i} {node} {1 / gp[d, i]:.17g}')
                lines.append(f'Rx{tag}m{d}{i} x{tag}m{i} {node} {1 / gm[d, i]:.17g}')
            columns.append((f'row_{tag}_{d}', f'v({node})'))
    return columns


def rows(port):
    q = np.array([port[f'row_q_{d}'][-1] for d in range(ROWS)])
    k = np.array([[port[f'row_k{j}_{d}'][-1] for d in range(ROWS)] for j in range(3)])
    return q, k


def dissipation(x_q, x_k, q, k, gp, gm, v_cm, scale):
    """Power dissipated in the crossbar resistors given solved row voltages."""
    total = 0.
    for x, r in [(x_q, q)] + [(x_k[j], k[j]) for j in range(3)]:
        for d in range(ROWS):
            for i in range(INPUTS):
                total += gp[d, i] * (v_cm + scale * x[i] - r[d]) ** 2
                total += gm[d, i] * (v_cm - scale * x[i] - r[d]) ** 2
    return float(total)
