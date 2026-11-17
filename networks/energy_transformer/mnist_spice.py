"""Transistor-level inference netlist of the MNIST chain A network (on-chip-learning #63).

One transient per image:
  16 tokens (7x7 patch pixels +-1 and a position one-hot) are applied in sequence through a
  passive resistor crossbar (d_in = 65 inputs -> d_k key + d_v value columns); transmission gates
  sample each token's keys and values onto hold capacitors (the KV cache), plus a V_cm reference.
  Per head: programmed query voltages (ideal sources), anti-bump cells per key and dimension
  (tail gating, summing-mirror bias from #62), capacitor integration, a PMOS Elfadel-Wyatt softmax
  into the attention controllers of a pseudoconductance bank with d_v features whose value
  anchors are fed by PMOS diff pairs on the held values. Outputs: read-mirror currents of the
  bank output nodes; the linear readout is digital.
Devices and sizes follow end_to_end.py chain A at the #59 L5 design.
"""
from pathlib import Path
import subprocess
import time
import numpy as np
from networks.energy_transformer import pseudoconductance as pc
from networks.energy_transformer import end_to_end as e2e
from networks.energy_transformer.end_to_end import pwl, pulse_high, EDGE, UT, VDD, IREF, H0

W_N, W_P, L, W_SW, L_SW, W_DP = e2e.W_N, e2e.W_P, e2e.L, e2e.W_SW, e2e.L_SW, e2e.W_DP


def timing(spec, n):
    tw = float(spec['t_write'])
    t_int0 = n * tw + float(spec['t_reset'])
    t_int1 = t_int0 + float(spec['t_integrate'])
    return {'t_write': tw, 't_int0': t_int0, 't_int1': t_int1, 't_end': t_int1 + float(spec['t_settle']),
            'sample': [(j * tw + .1 * tw, (j + .9) * tw) for j in range(n)], 'ref': ((n - 1) * tw + .1 * tw, (n - .1) * tw)}


def netlist(spec, model='bsim3'):
    x = np.asarray(spec['tokens'], float)                       # (N, d_in)
    kgp, kgm = np.asarray(spec['key_gp'], float), np.asarray(spec['key_gm'], float)     # (d_k, d_in)
    vgp, vgm = np.asarray(spec['value_gp'], float), np.asarray(spec['value_gm'], float)  # (d_v, d_in)
    qv = np.asarray(spec['query_V'], float)                     # (H, d_k)
    T_int = np.asarray(spec['t_integrate_heads'], float)        # (H,)
    N, d_in = x.shape
    d_k, d_v, H = kgp.shape[0], vgp.shape[0], qv.shape[0]
    spec = dict(spec, t_integrate=float(T_int.max()))
    T = timing(spec, N)
    v_cm, s, S, tail = float(spec['v_cm']), float(spec['scale']), float(spec['s_ratio']), float(spec['tail_current'])
    amp, vlow, cs, ch = float(spec['amplitude']), float(spec['v_low']), float(spec['c_score']), float(spec['c_hold'])
    prefix = 'M' if model == 'bsim3' else 'X'
    lines = ['MNIST chain A attention network',
             f'.options reltol={float(spec.get("reltol", 1e-6)):.3g} abstol={float(spec.get("abstol", 3e-15)):.3g} vntol=1e-9 gmin=1e-18 method=gear maxord=2',
             '.temp 27', f'VDD vdd 0 {VDD}', f'VREF vref 0 {pc.VREF}', f'VCMV vcmv 0 {v_cm:.17g}', f'Vlow vlow 0 {vlow:.17g}',
             f'Vread vrd 0 {pc.VREAD}', *pc.model_lines(model)]
    count, columns = {}, []

    def mos(name, d, g, s_, b, kind, group, width=None, length=L):
        if width is None:
            width = W_P if kind == 'pc' else W_N
        lines.append(f'{prefix}{name} {d} {g} {s_} {b} {kind} w={width:.17g} l={length:.17g}')
        count[group] = count.get(group, 0) + 1

    if model == 'ideal':
        lines.append('.model swm sw vt=0.9 vh=0.05 ron=1e3 roff=1e16')
        lines.append('.model swen sw vt=0.9 vh=0.05 ron=1e3 roff=1e12')

    def switch(name, a, b, clk, clkb):
        if model == 'ideal':
            lines.append(f'S{name} {a} {b} {clk} 0 swm')
        else:
            mos(f'{name}n', a, clk, b, '0', 'nc', 'switches', W_SW, L_SW)
            mos(f'{name}p', a, clkb, b, 'vdd', 'pc', 'switches', W_SW, L_SW)

    # token DAC (duty-cycled: V_cm after the last window) and crossbar
    for i in range(d_in):
        for sign, tag in ((1, 'p'), (-1, 'm')):
            pts = []
            for j in range(N):
                v = v_cm + sign * s * x[j, i]
                pts += [(j * T['t_write'] + (EDGE if j else 0), v), ((j + 1) * T['t_write'], v)]
            pts += [(N * T['t_write'] + EDGE, v_cm), (T['t_end'], v_cm)]
            lines.append(f'Vx{tag}{i} x{tag}{i} 0 {pwl(pts)}')
    cols = [(f'kc{d}', kgp[d], kgm[d]) for d in range(d_k)] + [(f'vc{e}', vgp[e], vgm[e]) for e in range(d_v)]
    for node, gp, gm in cols:
        for i in range(d_in):
            lines.append(f'Rx{node}p{i} xp{i} {node} {1 / gp[i]:.17g}')
            lines.append(f'Rx{node}m{i} xm{i} {node} {1 / gm[i]:.17g}')
    count['crossbar_resistors'] = 2 * d_in * len(cols)

    # KV cache and V_cm reference hold
    def hold(src, dst, clk, clkb):
        switch(f'sh_{dst}', src, dst, clk, clkb)
        lines.append(f'Chold_{dst} {dst} 0 {ch:.17g}')
        lines.append(f'.ic v({dst})={v_cm:.17g}')
        if spec.get('debug'):
            columns.append((f'hold_{dst}', f'v({dst})'))
    for j, (t0, t1) in enumerate(T['sample']):
        lines.append(f'Vphi{j} phi{j} 0 {pulse_high(t0, t1)}')
        lines.append(f'Vphib{j} phib{j} 0 {pulse_high(t0, t1, hi=0., lo=VDD)}')
        for d in range(d_k):
            hold(f'kc{d}', f'k{j}_{d}', f'phi{j}', f'phib{j}')
        for e in range(d_v):
            hold(f'vc{e}', f'v{j}_{e}', f'phi{j}', f'phib{j}')
    tr0, tr1 = T['ref']
    lines.append(f'Vphir phir 0 {pulse_high(tr0, tr1)}')
    lines.append(f'Vphibr phibr 0 {pulse_high(tr0, tr1, hi=0., lo=VDD)}')
    hold('vcmv', 'vcmh', 'phir', 'phibr')
    count['hold_capacitors'] = N * (d_k + d_v) + 1

    # shared clocks and references
    ta = max(T['t_int0'] - float(spec['t_warm']), EDGE)
    lines.append(f'Vrst rst 0 {pulse_high(0, T["t_int0"])}')
    lines.append(f'Vrstb rstb 0 {pulse_high(0, T["t_int0"], hi=0., lo=VDD)}')
    lines.append(f'Iref_tail vdd tref {tail:.17g}')
    mos('tref', 'tref', 'tref', '0', '0', 'nc', 'scorer')
    lines.append('.nodeset v(tref)=.6')
    lines.append(f'Iref_h vdd g_h {IREF:.17g}')
    mos('bias_h', 'g_h', 'g_h', 'vref', '0', 'nc', 'bank')
    lines.append(f'Iref_h0 vdd g_h0 {IREF * H0:.17g}')
    mos('bias_h0', 'g_h0', 'g_h0', 'vref', '0', 'nc', 'bank')
    lines.append(f'Iref_dp dref 0 {2 * IREF:.17g}')
    mos('dref', 'dref', 'dref', 'vdd', 'vdd', 'pc', 'value_conversion', W_DP)
    lines.append(f'Iref_ew ewr 0 {amp * IREF:.17g}')
    mos('ewr', 'ewr', 'ewr', 'vdd', 'vdd', 'pc', 'softmax')
    lines.append('.nodeset v(ewr)=1.1 v(g_h)=.7 v(g_h0)=.7')
    ib = float(spec.get('mirror_bias', 1e-9))
    gate = bool(spec.get('gate_scorer', True))

    for h in range(H):
        t1h = T['t_int0'] + T_int[h]                              # per-head integration window (temperature)
        if gate:
            lines.append(f'Ven{h} en{h} 0 {pulse_high(ta, t1h + EDGE)}')
        lines.append(f'Vint{h} int{h} 0 {pulse_high(T["t_int0"], t1h)}')
        lines.append(f'Vintb{h} intb{h} 0 {pulse_high(T["t_int0"], t1h, hi=0., lo=VDD)}')
        for d in range(d_k):
            lines.append(f'Vq{h}_{d} q{h}_{d} 0 {qv[h, d]:.17g}')
        mos(f'ewt{h}', f'ews{h}', 'ewr', 'vdd', 'vdd', 'pc', 'softmax')
        lines.append(f'.nodeset v(ews{h})=1.4')
        for j in range(N):
            tag = f'{h}_{j}'
            for d in range(d_k):
                t, xx = f't{tag}_{d}', f'x{tag}_{d}'
                if gate:
                    mos(f'en_{tag}_{d}', f'ts{tag}_{d}', f'en{h}', '0', '0', 'nc', 'switches', W_SW, L_SW)
                    mos(f'tail_{tag}_{d}', t, 'tref', f'ts{tag}_{d}', '0', 'nc', 'scorer')
                else:                                  # cells always on (no tail gating)
                    mos(f'tail_{tag}_{d}', t, 'tref', '0', '0', 'nc', 'scorer')
                mos(f'oa_{tag}_{d}', f'sum{tag}', f'q{h}_{d}', t, '0', 'nc', 'scorer')
                mos(f'ob_{tag}_{d}', f'sum{tag}', f'k{j}_{d}', t, '0', 'nc', 'scorer')
                mos(f'ma_{tag}_{d}', xx, f'q{h}_{d}', t, '0', 'nc', 'scorer', width=S * W_N)
                mos(f'mb_{tag}_{d}', 'vdd', f'k{j}_{d}', xx, '0', 'nc', 'scorer', width=S * W_N)
                lines.append(f'.nodeset v({t})=.45 v({xx})=.5 v(ts{tag}_{d})=.3')
            lines.append(f'Vsc_{tag} psum{tag} sum{tag} 0')
            # gated mirror bias (#62) on a small standing floor: an unbiased diode leaves the summing node floating
            ibf = float(spec.get('mirror_bias_floor', 1e-11))
            if gate:
                lines.append(f'Ibias_{tag} psum{tag} 0 ' + pwl([(0, ibf), (ta, ibf), (ta + EDGE, ib), (t1h + EDGE, ib), (t1h + 2 * EDGE, ibf)]))
            elif ib > 0:
                lines.append(f'Ibias_{tag} psum{tag} 0 {ib:.17g}')
            mos(f'pd_{tag}', f'psum{tag}', f'psum{tag}', 'vdd', 'vdd', 'pc', 'scorer', width=W_N)
            mos(f'pc_{tag}', f'pint{tag}', f'psum{tag}', 'vdd', 'vdd', 'pc', 'scorer', width=W_N)
            switch(f'int_{tag}', f'pint{tag}', f'gsc{tag}', f'int{h}', f'intb{h}')
            if spec.get('score_offset') is not None:   # programmed sink: removes the bias and a per-head offset,
                io = float(spec['score_offset'][h]) + (ib if gate else 0.)   # so only key differences integrate
                lines.append(f'Ioff_{tag} gsc{tag} 0 ' + pwl([(0, 0), (T['t_int0'], 0), (T['t_int0'] + EDGE, io), (t1h, io), (t1h + EDGE, 0)]))
            switch(f'dump_{tag}', f'pint{tag}', 'vlow', f'intb{h}', f'int{h}')
            switch(f'rst_{tag}', f'gsc{tag}', 'vlow', 'rst', 'rstb')
            lines.append(f'Cscore_{tag} gsc{tag} 0 {cs:.17g}')
            lines.append(f'.ic v(gsc{tag})={vlow:.17g}')
            lines.append(f'.nodeset v(psum{tag})=1.2 v(pint{tag})={vlow:.17g}')
            # softmax output into this key's attention controller
            lines.append(f'Vew_{tag} ewd{tag} ga{tag} 0')
            mos(f'ew_{tag}', f'ewd{tag}', f'gsc{tag}', f'ews{h}', 'vdd', 'pc', 'softmax')
            mos(f'bias_a{tag}', f'ga{tag}', f'ga{tag}', 'vref', '0', 'nc', 'bank')
            lines.append(f'.nodeset v(ga{tag})=.7')
            columns.append((f'softmax_{tag}', f'i(Vew_{tag})'))
            if spec.get('debug'):
                columns += [(f'gsc_{tag}', f'v(gsc{tag})'), (f'isum_{tag}', f'i(Vsc_{tag})')]
        # bank: one network per value feature, N value nodes + 1 output node
        for e in range(d_v):
            for j in range(N + 1):
                tag, node = f'{h}_{j}_{e}', f'z{h}_{j}_{e}'
                if j < N:
                    mos(f'dtail_{tag}', f'ds_{tag}', 'dref', 'vdd', 'vdd', 'pc', 'value_conversion', W_DP)
                    mos(f'dfeed_{tag}', f'data_{tag}', 'vcmh', f'ds_{tag}', 'vdd', 'pc', 'value_conversion', W_DP)
                    mos(f'ddump_{tag}', '0', f'v{j}_{e}', f'ds_{tag}', 'vdd', 'pc', 'value_conversion', W_DP)
                    gate = 'g_h'
                else:
                    lines.append(f'Ivalue_{tag} vdd data_{tag} {IREF * H0:.17g}')
                    gate = 'g_h0'
                mos(f'iref_{tag}', f'data_{tag}', f'data_{tag}', '0', '0', 'nc', 'bank')
                mos(f'isink_{tag}', node, f'data_{tag}', '0', '0', 'nc', 'bank')
                mos(f'pr_{tag}', f'sense_{tag}', f'sense_{tag}', 'vdd', 'vdd', 'pc', 'bank')
                mos(f'anchor_{tag}', f'sense_{tag}', gate, node, '0', 'nc', 'bank')
                lines.append(f'.ic v({node})=.3')
                lines.append(f'.nodeset v(data_{tag})=.3 v(sense_{tag})=1.5')
                if j == N:
                    mos(f'pcopy_{tag}', f'read_{tag}', f'sense_{tag}', 'vdd', 'vdd', 'pc', 'bank')
                    lines.append(f'Vread_{tag} read_{tag} 0 {pc.VREAD}')
                    columns.append((f'out_{h}_{e}', f'i(Vread_{tag})'))
            for j in range(N):
                mos(f'edge_{h}_{j}_{e}', f'z{h}_{j}_{e}', f'ga{h}_{j}', f'z{h}_{N}_{e}', '0', 'nc', 'bank')
    columns.append(('supply', 'i(VDD)'))
    tmax = min(T['t_end'] / 400, 2e-4)
    lines += ['.control', 'set noaskquit', 'set numdgt=12', 'set wr_vecnames', 'set wr_singlescale',
              f'tran {T["t_write"] / 200:.12g} {T["t_end"]:.12g} 0 {tmax:.12g}',
              'wrdata trace.txt ' + ' '.join(c_ for _, c_ in columns), 'quit', '.endc', '.end']
    return '\n'.join(lines) + '\n', columns, count


def simulate(spec, directory, executable, model='bsim3', timeout=7200):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    net, columns, count = netlist(spec, model)
    (directory / 'circuit.cir').write_text(net)
    t0 = time.monotonic()
    proc = subprocess.run([str(executable), '-b', 'circuit.cir'], cwd=directory, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - t0
    (directory / 'spice.log').write_text((proc.stdout + proc.stderr)[-20000:])
    if proc.returncode or not (directory / 'trace.txt').exists() or 'aborted' in (proc.stdout + proc.stderr).lower():
        raise RuntimeError(f'SPICE failed in {directory}: {(proc.stdout + proc.stderr)[-1500:]}')
    data = np.loadtxt(directory / 'trace.txt', skiprows=1, ndmin=2)
    if data.shape[1] != len(columns) + 1 or not np.isfinite(data).all():
        raise RuntimeError(f'Invalid trace {data.shape} in {directory}')
    port = {name: data[:, k + 1] for k, (name, _) in enumerate(columns)}
    H, N = len(spec['query_V']), len(spec['tokens'])
    d_v = len(spec['value_gp'])
    out = np.array([[port[f'out_{h}_{e}'][-1] / (IREF * H0) - 1 for e in range(d_v)] for h in range(H)])
    p = np.array([[port[f'softmax_{h}_{j}'][-1] for j in range(N)] for h in range(H)]) / (float(spec['amplitude']) * IREF)
    t = data[:, 0]
    i0 = int(np.searchsorted(t, .9 * t[-1]))
    drift = float(max(abs(port[f'out_{h}_{e}'][-1] - port[f'out_{h}_{e}'][i0]) for h in range(H) for e in range(d_v)) / (IREF * H0))
    np.savez_compressed(directory / 'trace_end.npz', **{k: v[-1:] for k, v in port.items()})
    (directory / 'trace.txt').unlink()                     # full waveforms are large; final values are kept
    if spec.get('debug'):
        Tm = timing(dict(spec, t_integrate=max(spec['t_integrate_heads'])), N)
        mids = [Tm['t_int0'] + .5 * ti for ti in spec['t_integrate_heads']]
        dbg = {k: float(np.interp(mids[int(k.split('_')[1])], t, v)) if k.startswith('isum') else float(v[-1])
               for k, v in port.items() if k.startswith(('gsc', 'isum', 'hold'))}
        return_debug = dbg
    else:
        return_debug = None
    return {'out': out, 'p': p, 'counts': count, 'seconds': elapsed, 'late_drift': drift, 'debug': return_debug,
            'supply_W_end': float(-VDD * port['supply'][-1])}
