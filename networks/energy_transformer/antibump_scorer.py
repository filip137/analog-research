"""Summed anti-bump attention scorer (on-chip-learning #57, stage 3).

Per key j and score dimension d, a Delbrueck bump-antibump cell: NMOS outer pair
(gates q_d, k_jd) sharing a tail with a series middle leg of two NMOS (gates q_d,
k_jd), each S times wider. The outer legs carry
    I_out = I_b * c cosh^2(u) / (1 + c cosh^2(u)),  c = 4/S,  u = kappa (q_d-k_jd)/(2 U_T)
and the middle leg the bump I_b / (1 + c cosh^2 u) (dumped to VDD). Outer drains of
all dimensions of key j sum (KCL) into a PMOS diode; its copy feeds a linear resistor
R, so the score voltage is linear in the summed dissimilarity current.

variant 'star'    : G_j = V_A - R * I_sum,j drives coupler gate g_a<j> directly
                    (NMOS sink mirror); V_A is a buffered controller-diode voltage
                    at amplitude*Iref, so a_j = amp * exp(-kappa R I_sum,j / U_T).
variant 'softmax' : G_j = V_low + R * I_sum,j drives PMOS Elfadel-Wyatt softmax
                    transistors with a shared tail current amp*Iref; their drains
                    feed the existing attention controllers, so
                    a_j = amp * softmax_j(-kappa R I_sum,j / U_T).
Ideal elements: query/key voltage sources, reference currents, R, the V_A buffer
(VCVS) and V_low. Zero-volt probes Vsc_<j> measure the summed score currents.
"""
import numpy as np

UT = 8.617333262145e-5 * 300.15
KEYS, DIMS = 3, 2
W_N, W_P, L = 10e-6, 10e-6, 10e-6


def law_currents(query, keys, s_ratio, kappa):
    """Outer-leg current / I_b per key and dimension."""
    c = 4.0 / s_ratio
    u = kappa * (np.asarray(query)[None, :] - np.asarray(keys)) / (2 * UT)
    ch = np.cosh(u) ** 2
    return c * ch / (1 + c * ch)


def law_attention(query, keys, spec, kappa):
    """Ideal attention currents a_j (units of Iref) for the given variant."""
    f = law_currents(query, keys, spec['s_ratio'], kappa).sum(1)
    expo = -kappa * spec['resistance'] * spec['tail_current'] * f / UT
    if spec['variant'] == 'star':
        return spec['amplitude'] * np.exp(expo)
    e = np.exp(expo - expo.max())
    return spec['amplitude'] * e / e.sum()


def add_scorer(lines, mos, spec):
    q = np.asarray(spec['query'], float)
    k = np.asarray(spec['keys'], float)
    if q.shape != (DIMS,) or k.shape != (KEYS, DIMS):
        raise ValueError('Expected a 2-D query and three 2-D keys')
    variant, S, tail = spec['variant'], float(spec['s_ratio']), float(spec['tail_current'])
    R, amp, iref = float(spec['resistance']), float(spec['amplitude']), float(spec['iref'])
    if variant not in ('star', 'softmax') or not 0 < tail <= 2e-9 or R <= 0:
        raise ValueError('Invalid anti-bump scorer specification')
    if (variant == 'star') != bool(spec.get('drives_gate')):
        raise ValueError('Star variant drives the coupler gates; softmax keeps the controllers')
    for d in range(DIMS):
        lines.append(f'Vq{d} q{d} 0 {q[d]:.17g}')
    for j in range(KEYS):
        for d in range(DIMS):
            lines.append(f'Vk{j}{d} k{j}{d} 0 {k[j, d]:.17g}')
    lines.append(f'Iref_tail vdd tref {tail:.17g}')
    mos('tref', 'tref', 'tref', '0', '0', 'nc', 'scorer')
    lines.append('.nodeset v(tref)=.6')
    columns = []
    for j in range(KEYS):
        for d in range(DIMS):
            t, x = f't{j}{d}', f'x{j}{d}'
            mos(f'tail_{j}{d}', t, 'tref', '0', '0', 'nc', 'scorer')
            mos(f'oa_{j}{d}', f'sum{j}', f'q{d}', t, '0', 'nc', 'scorer')
            mos(f'ob_{j}{d}', f'sum{j}', f'k{j}{d}', t, '0', 'nc', 'scorer')
            mos(f'ma_{j}{d}', x, f'q{d}', t, '0', 'nc', 'scorer', width=S * W_N)
            mos(f'mb_{j}{d}', 'vdd', f'k{j}{d}', x, '0', 'nc', 'scorer', width=S * W_N)
            lines.append(f'.nodeset v({t})=.45 v({x})=.5')
        # summed outer-leg current: PMOS diode, probe, copy
        lines.append(f'Vsc_{j} psum{j} sum{j} 0')
        mos(f'pd_{j}', f'psum{j}', f'psum{j}', 'vdd', 'vdd', 'pc', 'scorer', width=W_P)
        lines.append(f'.nodeset v(psum{j})=1.2')
        columns.append((f'score_{j}', f'i(Vsc_{j})'))
        if variant == 'star':
            mos(f'pc_{j}', f'nm{j}', f'psum{j}', 'vdd', 'vdd', 'pc', 'scorer', width=W_P)
            mos(f'nd_{j}', f'nm{j}', f'nm{j}', '0', '0', 'nc', 'scorer')
            mos(f'ns_{j}', f'g_a{j}', f'nm{j}', '0', '0', 'nc', 'scorer')
            lines.append(f'R_score_{j} va_buf g_a{j} {R:.17g}')
            lines.append(f'.nodeset v(nm{j})=.55 v(g_a{j})=.75')
            columns.append((f'gate_score_{j}', f'v(g_a{j})'))
        else:
            mos(f'pc_{j}', f'gsc{j}', f'psum{j}', 'vdd', 'vdd', 'pc', 'scorer', width=W_P)
            lines.append(f'R_score_{j} gsc{j} vlow {R:.17g}')
            mos(f'ew_{j}', f'g_a{j}', f'gsc{j}', 'ews', 'vdd', 'pc', 'scorer')
            lines.append(f'.nodeset v(gsc{j})=.75')
            columns.append((f'gate_score_{j}', f'v(gsc{j})'))
    if variant == 'star':
        # amplitude reference: controller diode at amp*Iref, buffered (ideal VCVS)
        lines.append(f'Iref_va vdd va {amp * iref:.17g}')
        mos('va', 'va', 'va', 'vref', '0', 'nc', 'scorer')
        lines.append('Eva va_buf 0 va 0 1')
        lines.append('.nodeset v(va)=.8')
    else:
        lines.append(f'Vlow vlow 0 {float(spec["v_low"]):.17g}')
        lines.append(f'Iref_ew ewr 0 {amp * iref:.17g}')
        mos('ewr', 'ewr', 'ewr', 'vdd', 'vdd', 'pc', 'scorer')
        mos('ewt', 'ews', 'ewr', 'vdd', 'vdd', 'pc', 'scorer')
        lines.append('.nodeset v(ewr)=1.1 v(ews)=1.4')
        columns.append(('ew_source', 'v(ews)'))
    return columns


def measured_scores(port):
    return np.array([port[f'score_{j}'][-1] for j in range(KEYS)])


def score_gates(port):
    return np.array([port[f'gate_score_{j}'][-1] for j in range(KEYS)])


def sum_test_netlist(deltas, active, model, s_ratio, tail_current, v_cm=1.05):
    """Standalone summing test: one key, len(deltas) cells into one PMOS diode.
    Only cells whose index is in `active` are instantiated."""
    from networks.energy_transformer import pseudoconductance as pc
    lines = ['Anti-bump summation test', '.options reltol=1e-7 abstol=1e-20 vntol=1e-10 gmin=1e-18 method=gear maxord=2',
             '.temp 27', 'VDD vdd 0 1.8', *pc.model_lines(model)]
    prefix = 'M' if model == 'bsim3' else 'X'

    def mos(name, d, g, s, b, kind, width):
        lines.append(f'{prefix}{name} {d} {g} {s} {b} {kind} w={width:.17g} l={L:.17g}')
    lines.append(f'Iref_tail vdd tref {tail_current:.17g}')
    mos('tref', 'tref', 'tref', '0', '0', 'nc', W_N)
    lines.append(f'Vq q 0 {v_cm:.17g}')
    for i in active:
        lines.append(f'Vk{i} k{i} 0 {v_cm - deltas[i]:.17g}')
        mos(f'tail{i}', f't{i}', 'tref', '0', '0', 'nc', W_N)
        mos(f'oa{i}', 'sum', 'q', f't{i}', '0', 'nc', W_N)
        mos(f'ob{i}', 'sum', f'k{i}', f't{i}', '0', 'nc', W_N)
        mos(f'ma{i}', f'x{i}', 'q', f't{i}', '0', 'nc', s_ratio * W_N)
        mos(f'mb{i}', 'vdd', f'k{i}', f'x{i}', '0', 'nc', s_ratio * W_N)
        lines.append(f'.nodeset v(t{i})=.45 v(x{i})=.5')
    lines.append('.nodeset v(tref)=.6 v(psum)=1.3')
    lines.append('Vsc psum sum 0')
    mos('pd', 'psum', 'psum', 'vdd', 'vdd', 'pc', W_P)
    lines += ['.control', 'set noaskquit', 'set numdgt=15', 'set wr_vecnames', 'set wr_singlescale', 'op',
              'wrdata trace.txt i(Vsc) v(psum)', 'quit', '.endc', '.end']
    return '\n'.join(lines) + '\n'


def run_sum_test(directory, executable, **kw):
    import subprocess, time
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'circuit.cir').write_text(sum_test_netlist(**kw))
    t0 = time.monotonic()
    proc = subprocess.run([str(executable), '-b', 'circuit.cir'], cwd=directory, capture_output=True, text=True, timeout=10)
    elapsed = time.monotonic() - t0
    (directory / 'spice.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode or not (directory / 'trace.txt').exists() or 'aborted' in (proc.stdout + proc.stderr).lower():
        raise RuntimeError(f'SPICE failed in {directory}')
    data = np.loadtxt(directory / 'trace.txt', skiprows=1, ndmin=2)
    if data.shape[1] != 3 or not np.isfinite(data).all():
        raise RuntimeError(f'Invalid trace in {directory}')
    return float(data[-1, 1]), float(data[-1, 2]), elapsed
