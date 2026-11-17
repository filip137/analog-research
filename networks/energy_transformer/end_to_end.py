"""End-to-end analog attention step (on-chip-learning #58, stage 3).

One transient covers a whole attention step for three context tokens and one query:

1. Token sequence -> passive differential resistive crossbar (#56): each token
   x = [value, kx, ky] is applied in turn as V_cm +/- s*x_i (ideal PWL sources, the
   token DAC). Unloaded columns give key voltages kc0, kc1 and the value voltage vc
   as weighted means.
2. KV cache: during token j's window, transmission gates (BSIM3) or ideal switches
   copy kc0, kc1 and vc onto hold capacitors k<j>0, k<j>1 and v<j>.
3. Query: the query token is applied; kc0, kc1 drive the scorer's query gates live.
4. Scorer: 5-MOS Delbrueck anti-bump cells per key and dimension (#57), summed per
   key into a PMOS diode; a PMOS copy is integrated on a score capacitor C for a
   window T after a reset to V_low (capacitor integration replaces the ideal
   resistor of #57): V_j = V_low + T * I_sum,j / C.
5. Elfadel-Wyatt PMOS softmax on the held score voltages (shared tail amp*Iref).
6a. Chain A: softmax currents feed the attention controllers of the 51-MOS
   pseudoconductance bank; each stored value drives a PMOS diff pair whose branch
   current Iref*(1 +/- tanh(kappa (V_v - V_cm)/2U_T)) is the bank's value input
   (feature 0 carries +v, feature 1 -v). Output: read-mirror current of node z3.
6b. Chain B: each softmax current is mirrored into the tails of NMOS diff pairs
   whose gates hold the value; the differential output current is
   amp*Iref * sum_j p_j tanh(kappa (V_v,j - V_cm)/2U_T). No bank.
Ideal elements: token sources, reference currents, V_low, V_cm reference, output
probes/compliance sources, and (ideal model only) switches. Clock waveforms are
ideal PWL sources.
"""
from pathlib import Path
import subprocess
import time
import numpy as np
from networks.energy_transformer import pseudoconductance as pc
from networks.energy_transformer import antibump_scorer as ab

UT = pc.UT
VDD, VREF, VREAD, IREF, H0 = pc.VDD, pc.VREF, pc.VREAD, pc.IREF, pc.H0
W_N, W_P, L = 10e-6, 1e-6, 10e-6
W_SW, L_SW = 10e-6, 1e-6
W_DP = 20e-6                  # value diff-pair PMOS width (headroom at V_cm = 1.05 V)
EDGE = 100e-9
KEYS, DIMS, INPUTS = 3, 2, 3


def timing(spec):
    tw = float(spec['t_write'])
    t_q = KEYS * tw
    t_int0 = t_q + float(spec['t_reset'])
    t_int1 = t_int0 + float(spec['t_integrate'])
    return {'t_write': tw, 't_query': t_q, 't_int0': t_int0, 't_int1': t_int1,
            't_end': t_int1 + float(spec['t_settle']), 'sample': [(j * tw + .1 * tw, (j + .9) * tw) for j in range(KEYS)]}


def pwl(points):
    return 'PWL(' + ' '.join(f'{t:.12g} {v:.12g}' for t, v in points) + ')'


def pulse_high(t0, t1, hi=VDD, lo=0.):
    if t0 <= 0:
        return pwl([(0, hi), (t1, hi), (t1 + EDGE, lo)])
    return pwl([(0, lo), (t0, lo), (t0 + EDGE, hi), (t1, hi), (t1 + EDGE, lo)])


def law(spec, kappa):
    """Ideal-law prediction for the netlist: weighted-mean crossbar, anti-bump cells,
    linear integration, softmax, then bank (A) or diff-pair weighting (B)."""
    ctx, xq = np.asarray(spec['ctx'], float), np.asarray(spec['xq'], float)
    kw = (np.asarray(spec['key_gp']) - np.asarray(spec['key_gm'])) / (np.asarray(spec['key_gp']) + np.asarray(spec['key_gm'])).sum(-1, keepdims=True)
    vw = (np.asarray(spec['value_gp']) - np.asarray(spec['value_gm'])) / (np.asarray(spec['value_gp']) + np.asarray(spec['value_gm'])).sum(-1, keepdims=True)
    q = spec['v_cm'] + spec['scale'] * xq @ kw.T
    k = spec['v_cm'] + spec['scale'] * ctx @ kw.T
    dv = spec['scale'] * (ctx @ vw.T)[:, 0]
    f = ab.law_currents(q, k, spec['s_ratio'], kappa).sum(1)
    expo = -kappa * spec['t_integrate'] * spec['tail_current'] * f / (spec['c_score'] * UT)
    p = np.exp(expo - expo.max()); p /= p.sum()
    t = np.tanh(kappa * dv / (2 * UT))
    if spec['chain'] == 'A':
        a = spec['amplitude'] * p
        w = a / (1 + a)
        y = w @ t / (H0 + w.sum())
    else:
        y = p @ t
    return {'q': q, 'k': k, 'v': spec['v_cm'] + dv, 'isum': f * spec['tail_current'], 'p': p, 'y': y}


def netlist(spec, model):
    chain = spec['chain']
    if chain not in ('A', 'B') or model not in ('ideal', 'bsim3'):
        raise ValueError('chain A|B, model ideal|bsim3')
    ctx, xq = np.asarray(spec['ctx'], float), np.asarray(spec['xq'], float)
    kgp, kgm = np.asarray(spec['key_gp'], float), np.asarray(spec['key_gm'], float)
    vgp, vgm = np.asarray(spec['value_gp'], float), np.asarray(spec['value_gm'], float)
    if ctx.shape != (KEYS, INPUTS) or xq.shape != (INPUTS,) or kgp.shape != (DIMS, INPUTS) or vgp.shape != (1, INPUTS):
        raise ValueError('Expected 3 tokens of 3 inputs, a 2x3 key and 1x3 value crossbar')
    if min(kgp.min(), kgm.min(), vgp.min(), vgm.min()) <= 0:
        raise ValueError('Conductances must be positive')
    T = timing(spec)
    v_cm, s, S, tail = float(spec['v_cm']), float(spec['scale']), float(spec['s_ratio']), float(spec['tail_current'])
    amp, vlow, cs, ch = float(spec['amplitude']), float(spec['v_low']), float(spec['c_score']), float(spec['c_hold'])
    prefix = 'M' if model == 'bsim3' else 'X'
    lines = [f'End-to-end analog attention step, chain {chain}',
             f'.options reltol=1e-6 abstol={float(spec.get("abstol", 1e-16)):.3g} vntol=1e-9 gmin=1e-18 method=gear maxord=2',
             '.temp 27', f'VDD vdd 0 {VDD}', f'VREF vref 0 {VREF}', f'VCMV vcmv 0 {v_cm:.17g}', f'Vlow vlow 0 {vlow:.17g}',
             *pc.model_lines(model)]
    count = {}
    columns = []

    def mos(name, d, g, s_, b, kind, group, width=None, length=L):
        if width is None:
            width = W_P if kind == 'pc' else W_N
        lines.append(f'{prefix}{name} {d} {g} {s_} {b} {kind} w={width:.17g} l={length:.17g}')
        count[group] = count.get(group, 0) + 1

    if model == 'ideal':
        lines.append('.model swm sw vt=0.9 vh=0.05 ron=1e3 roff=1e16')

    def switch(name, a, b, clk, clkb):
        """Transmission gate (BSIM3) or ideal switch, closed while clk is high."""
        if model == 'ideal':
            lines.append(f'S{name} {a} {b} {clk} 0 swm')
        else:
            mos(f'{name}n', a, clk, b, '0', 'nc', 'switches', W_SW, L_SW)
            mos(f'{name}p', a, clkb, b, 'vdd', 'pc', 'switches', W_SW, L_SW)

    # 1. token DAC and crossbar
    seq = [ctx[j] for j in range(KEYS)] + [xq]
    for i in range(INPUTS):
        for sign, tag in ((1, 'p'), (-1, 'm')):
            pts = []
            for j, x in enumerate(seq):
                v = v_cm + sign * s * x[i]
                t0 = j * T['t_write']
                pts += [(t0 + (EDGE if j else 0), v), ((j + 1) * T['t_write'] if j < KEYS else T['t_end'], v)]
            lines.append(f'Vx{tag}{i} x{tag}{i} 0 {pwl(pts)}')
    cols = [('kc0', kgp[0], kgm[0]), ('kc1', kgp[1], kgm[1]), ('vc', vgp[0], vgm[0])]
    for node, gp, gm in cols:
        for i in range(INPUTS):
            lines.append(f'Rx{node}p{i} xp{i} {node} {1 / gp[i]:.17g}')
            lines.append(f'Rx{node}m{i} xm{i} {node} {1 / gm[i]:.17g}')
        columns.append((f'col_{node}', f'v({node})'))
    count['crossbar_resistors'] = 6 * len(cols)

    # 2. KV cache: sample-and-hold per token slot
    for j, (t0, t1) in enumerate(T['sample']):
        lines.append(f'Vphi{j} phi{j} 0 {pulse_high(t0, t1)}')
        lines.append(f'Vphib{j} phib{j} 0 {pulse_high(t0, t1, hi=0., lo=VDD)}')
        for src, dst in (('kc0', f'k{j}0'), ('kc1', f'k{j}1'), ('vc', f'v{j}')):
            switch(f'sh_{dst}', src, dst, f'phi{j}', f'phib{j}')
            lines.append(f'Chold_{dst} {dst} 0 {ch:.17g}')
            lines.append(f'.ic v({dst})={v_cm:.17g}')
            columns.append((f'hold_{dst}', f'v({dst})'))
    count['hold_capacitors'] = 3 * KEYS

    # 4. anti-bump scorer with capacitor integration
    lines.append(f'Iref_tail vdd tref {tail:.17g}')
    mos('tref', 'tref', 'tref', '0', '0', 'nc', 'scorer')
    lines.append('.nodeset v(tref)=.6')
    lines.append(f'Vrst rst 0 {pulse_high(0, T["t_int0"])}')
    lines.append(f'Vint int 0 {pulse_high(T["t_int0"], T["t_int1"])}')
    lines.append(f'Vintb intb 0 {pulse_high(T["t_int0"], T["t_int1"], hi=0., lo=VDD)}')
    for j in range(KEYS):
        for d in range(DIMS):
            t, x = f't{j}{d}', f'x{j}{d}'
            mos(f'tail_{j}{d}', t, 'tref', '0', '0', 'nc', 'scorer')
            mos(f'oa_{j}{d}', f'sum{j}', f'kc{d}', t, '0', 'nc', 'scorer')
            mos(f'ob_{j}{d}', f'sum{j}', f'k{j}{d}', t, '0', 'nc', 'scorer')
            mos(f'ma_{j}{d}', x, f'kc{d}', t, '0', 'nc', 'scorer', width=S * W_N)
            mos(f'mb_{j}{d}', 'vdd', f'k{j}{d}', x, '0', 'nc', 'scorer', width=S * W_N)
            lines.append(f'.nodeset v({t})=.45 v({x})=.5')
        lines.append(f'Vsc_{j} psum{j} sum{j} 0')
        mos(f'pd_{j}', f'psum{j}', f'psum{j}', 'vdd', 'vdd', 'pc', 'scorer', width=W_N)
        mos(f'pc_{j}', f'pint{j}', f'psum{j}', 'vdd', 'vdd', 'pc', 'scorer', width=W_N)
        switch(f'int_{j}', f'pint{j}', f'gsc{j}', 'int', 'intb')
        switch(f'dump_{j}', f'pint{j}', 'vlow', 'intb', 'int')
        switch(f'rst_{j}', f'gsc{j}', 'vlow', 'rst', f'rstb')
        lines.append(f'.nodeset v(psum{j})=1.2 v(pint{j})={vlow:.17g}')
        lines.append(f'Cscore_{j} gsc{j} 0 {cs:.17g}')
        lines.append(f'.ic v(gsc{j})={vlow:.17g}')
        columns += [(f'score_{j}', f'i(Vsc_{j})'), (f'gate_score_{j}', f'v(gsc{j})')]
    lines.append(f'Vrstb rstb 0 {pulse_high(0, T["t_int0"], hi=0., lo=VDD)}')

    # 5. Elfadel-Wyatt softmax
    lines.append(f'Iref_ew ewr 0 {amp * IREF:.17g}')
    mos('ewr', 'ewr', 'ewr', 'vdd', 'vdd', 'pc', 'softmax')
    mos('ewt', 'ews', 'ewr', 'vdd', 'vdd', 'pc', 'softmax')
    lines.append('.nodeset v(ewr)=1.1 v(ews)=1.4')
    for j in range(KEYS):
        lines.append(f'Vew_{j} ewd{j} g_a{j} 0')
        mos(f'ew_{j}', f'ewd{j}', f'gsc{j}', 'ews', 'vdd', 'pc', 'softmax')
        columns.append((f'softmax_{j}', f'i(Vew_{j})'))

    if chain == 'A':
        _bank(lines, mos, columns, ch)
    else:
        _translinear(lines, mos, columns)
    columns.append(('supply', 'i(VDD)'))
    tmax = min(T['t_end'] / 400, 2e-4)
    lines += ['.control', 'set noaskquit', 'set numdgt=15', 'set wr_vecnames', 'set wr_singlescale',
              f'tran {T["t_write"] / 200:.12g} {T["t_end"]:.12g} 0 {tmax:.12g}',
              'wrdata trace.txt ' + ' '.join(e for _, e in columns), 'quit', '.endc', '.end']
    return '\n'.join(lines) + '\n', columns, count


def _bank(lines, mos, columns, ch):
    """51-MOS bank of pseudoconductance.export with diff-pair value inputs."""
    for label, current in (('h', IREF), ('h0', IREF * H0)):
        lines.append(f'Iref_{label} vdd g_{label} {current:.17g}')
        mos('bias_' + label, 'g_' + label, 'g_' + label, 'vref', '0', 'nc', 'bank')
    for j in range(KEYS):
        mos(f'bias_a{j}', f'g_a{j}', f'g_a{j}', 'vref', '0', 'nc', 'bank')
        lines.append(f'.nodeset v(g_a{j})=.7')
    lines.append(f'Iref_dp dref 0 {2 * IREF:.17g}')
    mos('dref', 'dref', 'dref', 'vdd', 'vdd', 'pc', 'value_conversion', W_DP)
    for d in range(DIMS):
        for j in range(KEYS + 1):
            tag, node = f'{j}_{d}', f'z{j}_{d}'
            gate = 'g_h' if j < KEYS else 'g_h0'
            if j < KEYS:      # PMOS diff pair: branch with gate V_cm feeds the input mirror
                mos(f'dtail_{tag}', f'ds_{tag}', 'dref', 'vdd', 'vdd', 'pc', 'value_conversion', W_DP)
                feed, dump = ('vcmv', f'v{j}') if d == 0 else (f'v{j}', 'vcmv')
                lines.append(f'Vdin_{tag} dfeed_{tag} data_{tag} 0')
                mos(f'dfeed_{tag}', f'dfeed_{tag}', feed, f'ds_{tag}', 'vdd', 'pc', 'value_conversion', W_DP)
                mos(f'ddump_{tag}', '0', dump, f'ds_{tag}', 'vdd', 'pc', 'value_conversion', W_DP)
                columns.append((f'value_in_{tag}', f'i(Vdin_{tag})'))
            else:
                lines.append(f'Ivalue_{tag} vdd data_{tag} {IREF * H0:.17g}')
            mos('iref_' + tag, 'data_' + tag, 'data_' + tag, '0', '0', 'nc', 'bank')
            mos('isink_' + tag, node, 'data_' + tag, '0', '0', 'nc', 'bank')
            mos('pr_' + tag, 'sense_' + tag, 'sense_' + tag, 'vdd', 'vdd', 'pc', 'bank')
            lines.append(f'Vanc_{tag} sense_{tag} anc_{tag} 0')
            mos('anchor_' + tag, 'anc_' + tag, gate, node, '0', 'nc', 'bank')
            mos('pcopy_' + tag, 'read_' + tag, 'sense_' + tag, 'vdd', 'vdd', 'pc', 'bank')
            lines.append(f'Vread_{tag} read_{tag} 0 {VREAD}')
            lines.append(f'Cstate_{tag} {node} 0 {pc.CAP:.17g}')
            lines.append(f'.ic v({node})=.3')
            lines.append(f'.nodeset v(data_{tag})=.3 v(sense_{tag})=1.5')
            columns += [(f'read_{tag}', f'i(Vread_{tag})'), (f'v_{tag}', f'v({node})'),
                        (f'sense_{tag}', f'v(sense_{tag})')]
        for j in range(KEYS):
            mos(f'edge_{j}_{d}', f'z{j}_{d}', f'g_a{j}', f'z3_{d}', '0', 'nc', 'bank')


def _translinear(lines, mos, columns):
    """Softmax currents as diff-pair tails; outputs into ideal compliance probes."""
    for d in range(DIMS):
        lines.append(f'Vop_{d} op{d} 0 {VREAD}')
        lines.append(f'Vom_{d} om{d} 0 {VREAD}')
        columns += [(f'out_p_{d}', f'i(Vop_{d})'), (f'out_m_{d}', f'i(Vom_{d})')]
    for j in range(KEYS):
        mos(f'bd_{j}', f'g_a{j}', f'g_a{j}', '0', '0', 'nc', 'translinear')
        for d in range(DIMS):
            tag = f'{j}_{d}'
            mos(f'bt_{tag}', f'bs_{tag}', f'g_a{j}', '0', '0', 'nc', 'translinear')
            vp, vm = (f'v{j}', 'vcmv') if d == 0 else ('vcmv', f'v{j}')
            mos(f'bp_{tag}', f'op{d}', vp, f'bs_{tag}', '0', 'nc', 'translinear')
            mos(f'bm_{tag}', f'om{d}', vm, f'bs_{tag}', '0', 'nc', 'translinear')


def decode(port, spec):
    T = timing(spec)
    t = port['time']
    out = {}
    if spec['chain'] == 'A':
        x = np.array([port[f'read_3_{d}'][-1] / (IREF * H0) for d in range(DIMS)])
        out['y_raw'] = x - 1
        out['v_node_range'] = [float(min(port[f'v_{j}_{d}'][-1] for j in range(4) for d in range(DIMS))),
                               float(max(port[f'v_{j}_{d}'][-1] for j in range(4) for d in range(DIMS)))]
        out['min_anchor_vds'] = float(min(port[f'sense_{j}_{d}'][-1] - port[f'v_{j}_{d}'][-1]
                                          for j in range(4) for d in range(DIMS)))
        # settling: change of the output over the last 10% of the hold
        i0 = np.searchsorted(t, t[-1] - .1 * (t[-1] - T['t_int1']))
        out['late_drift'] = float(max(abs(port[f'read_3_{d}'][-1] - port[f'read_3_{d}'][i0]) / (IREF * H0) for d in range(DIMS)))
    else:
        amp = float(spec['amplitude']) * IREF
        # probes measure current flowing out of the probe source into the drains
        out['y_raw'] = np.array([(port[f'out_m_{d}'][-1] - port[f'out_p_{d}'][-1]) / amp for d in range(DIMS)])
        out['late_drift'] = 0.
    out['softmax'] = np.array([port[f'softmax_{j}'][-1] for j in range(KEYS)]) / (float(spec['amplitude']) * IREF)
    out['isum_integrating'] = np.array([np.interp(.5 * (T['t_int0'] + T['t_int1']), t, port[f'score_{j}']) for j in range(KEYS)])
    out['score_v'] = np.array([port[f'gate_score_{j}'][-1] for j in range(KEYS)])
    # KV cache: column value just before the switch opens, held value just after, and at the end
    held = {}
    for j, (t0, t1) in enumerate(T['sample']):
        for name, col in ((f'k{j}0', 'kc0'), (f'k{j}1', 'kc1'), (f'v{j}', 'vc')):
            before = float(np.interp(t1 - EDGE, t, port[f'col_{col}']))
            after = float(np.interp(t1 + 5 * EDGE + .02 * T['t_write'], t, port[f'hold_{name}']))
            end = float(port[f'hold_{name}'][-1])
            held[name] = {'column': before, 'held': after, 'end': end}
    out['kv'] = held
    out['hold_seconds'] = float(t[-1] - T['sample'][-1][1])
    return out


def simulate(spec, model, directory, executable, timeout=600):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    net, columns, count = netlist(spec, model)
    (directory / 'circuit.cir').write_text(net)
    t0 = time.monotonic()
    proc = subprocess.run([str(executable), '-b', 'circuit.cir'], cwd=directory, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - t0
    (directory / 'spice.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode or not (directory / 'trace.txt').exists() or 'aborted' in (proc.stdout + proc.stderr).lower():
        raise RuntimeError(f'SPICE failed in {directory}: {(proc.stdout + proc.stderr)[-1500:]}')
    data = np.loadtxt(directory / 'trace.txt', skiprows=1, ndmin=2)
    if data.shape[1] != len(columns) + 1 or not np.isfinite(data).all():
        raise RuntimeError(f'Invalid trace {data.shape} in {directory}')
    if data[-1, 0] < timing(spec)['t_end'] * (1 - 1e-6):
        raise RuntimeError(f'Incomplete transient in {directory}')
    port = {'time': data[:, 0], **{name: data[:, k + 1] for k, (name, _) in enumerate(columns)}}
    np.savez_compressed(directory / 'trace.npz', **port)
    res = decode(port, spec)
    supply = port['supply']
    res['supply_power_W_end'] = float(-VDD * supply[-1])
    res['counts'] = count
    res['seconds'] = elapsed
    return res
