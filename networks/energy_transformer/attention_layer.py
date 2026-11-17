"""Analog attention layer end-to-end (on-chip-learning #61): chain B attention, L1 layer norm,
DRN MLP, in one transient.

Front end (as end_to_end.py chain B with the #59/#60 options): tokens x = [v1, v2, kx, ky]
applied in sequence to an explicit resistor crossbar with two key and two value columns;
transmission-gate sample-and-hold of keys, values, the query and a V_cm reference; anti-bump
cells (1 um, S = 4) with tail gating and capacitor integration; PMOS Elfadel-Wyatt softmax;
the softmax currents are mirrored into the tails of NMOS diff pairs that hold each value
feature, giving per feature e the currents I_p,e and I_m,e with
    I_p,e - I_m,e = I_tail * sum_j p_j tanh(kappa (V_v,je - V_cm) / 2U_T) = I_tail * o_e.

Simple layer norm (L1, ScaleNorm-type, no centring), subthreshold current mode:
  rectifier: PMOS copies place +I_p,e and -I_m,e on node r_e; an NMOS diode to ground
  carries D+ = max(D, 0) and a PMOS diode from a low rail V_rect carries D- = max(-D, 0)
  (the rail leaves a dead band, so the two never conduct together); a PMOS copy of D- feeds
  an NMOS diode.
  normalizer: the four NMOS diodes (D+_0, D-_0, D+_1, D-_1) plus one fed by I_eps drive
  equal NMOS transistors sharing a source node with tail I_T, so output_i = I_T * I_i / sum.
  Hence n_e = (out+_e - out-_e) / I_T = o_e / (|o|_1 + eps), eps = I_eps / I_tail.
Interface (ideal, labelled): behavioural sources convert the normalized output currents into
DRN input voltages u = V_ref + V_U [n0, n1, -n0, -n1, +1, -1] (transimpedance plus buffer).
DRN (pseudo_drn.py): positive resistors g1 (6 x H) to hidden nodes clamped >= V_ref by
perfect diodes (behavioural, softplus with a 1 mV corner), shunts G_s to V_ref, resistors g2
(H x C) to output nodes with shunts. Class = argmax of output voltages (software).
"""
from pathlib import Path
import subprocess
import time
import numpy as np
from networks.energy_transformer import pseudoconductance as pc
from networks.energy_transformer import end_to_end as e2e
from networks.energy_transformer.end_to_end import pwl, pulse_high, EDGE, UT, VDD, IREF

KEYS, KDIMS, VDIMS, INPUTS = 3, 2, 2, 4
W_N, W_P, L = e2e.W_N, e2e.W_P, e2e.L
W_SW, L_SW = e2e.W_SW, e2e.L_SW


def timing(spec):
    tw = float(spec['t_write'])
    t_q = KEYS * tw
    t_int0 = t_q + tw + float(spec['t_reset'])
    t_int1 = t_int0 + float(spec['t_integrate'])
    return {'t_write': tw, 't_query': t_q, 't_int0': t_int0, 't_int1': t_int1, 't_end': t_int1 + float(spec['t_settle']),
            'sample': [(j * tw + .1 * tw, (j + .9) * tw) for j in range(KEYS)],
            'query': (t_q + .1 * tw, t_q + .9 * tw)}


def weights(gp, gm):
    gp, gm = np.asarray(gp, float), np.asarray(gm, float)
    return (gp - gm) / (gp + gm).sum(-1, keepdims=True)


def law_attention(spec, kappa_cell, s_eff, kappa):
    """Equation model of the front end (fitted cell law): returns o (2,), p (3,)."""
    ctx, xq = np.asarray(spec['ctx'], float), np.asarray(spec['xq'], float)
    wk, wv = weights(spec['key_gp'], spec['key_gm']), weights(spec['value_gp'], spec['value_gm'])
    q = spec['v_cm'] + spec['scale'] * xq @ wk.T
    k = spec['v_cm'] + spec['scale'] * ctx @ wk.T
    c = 4 / s_eff
    ch = np.cosh(kappa_cell * (q[None] - k) / (2 * UT)) ** 2
    f = (c * ch / (1 + c * ch)).sum(1)
    s = -spec['temperature_g_trained'] * f
    p = np.exp(s - s.max()); p /= p.sum()
    dv = spec['scale'] * ctx @ wv.T
    o = (p[:, None] * np.tanh(kappa * dv / (2 * UT))).sum(0)
    return o, p


def law_norm(o, eps):
    return np.asarray(o) / (np.abs(o).sum() + eps)


def law_drn(u, g1, g2, g_s, iters=400):
    g1, g2 = np.asarray(g1), np.asarray(g2)
    dh, do = g_s + g1.sum(0) + g2.sum(1), g_s + g2.sum(0)
    o = np.zeros(g2.shape[1])
    for _ in range(iters):
        h = np.maximum((u @ g1 + g2 @ o) / dh, 0)
        o = (h @ g2) / do
    return h, o


def drn_inputs(n, v_u):
    n = np.asarray(n)
    return v_u * np.concatenate([n, -n, [1., -1.]])


def netlist(spec, model):
    if model not in ('ideal', 'bsim3'):
        raise ValueError(model)
    ctx, xq = np.asarray(spec['ctx'], float), np.asarray(spec['xq'], float)
    kgp, kgm = np.asarray(spec['key_gp'], float), np.asarray(spec['key_gm'], float)
    vgp, vgm = np.asarray(spec['value_gp'], float), np.asarray(spec['value_gm'], float)
    g1, g2 = np.asarray(spec['drn_g1'], float), np.asarray(spec['drn_g2'], float)
    if ctx.shape != (KEYS, INPUTS) or xq.shape != (INPUTS,) or kgp.shape != (KDIMS, INPUTS) or vgp.shape != (VDIMS, INPUTS):
        raise ValueError('Expected 3 tokens of 4 inputs, 2x4 key and 2x4 value crossbars')
    if g1.shape[0] != 6 or g2.shape[0] != g1.shape[1]:
        raise ValueError('DRN shapes: g1 (6, H), g2 (H, C)')
    T = timing(spec)
    v_cm, s, S, tail = float(spec['v_cm']), float(spec['scale']), float(spec['s_ratio']), float(spec['tail_current'])
    amp, vlow, cs, ch = float(spec['amplitude']), float(spec['v_low']), float(spec['c_score']), float(spec['c_hold'])
    geometry = spec.get('geometry') or {}
    i_t, i_eps = float(spec['norm_tail']), float(spec['eps']) * amp * IREF
    w_ln, l_ln = float(spec['norm_width']), float(spec['norm_length'])
    v_u, v_ref, g_s = float(spec['v_u']), float(spec['drn_ref']), float(spec['g_s'])
    prefix = 'M' if model == 'bsim3' else 'X'
    lines = ['Analog attention layer: chain B attention, L1 layer norm, DRN MLP',
             f'.options reltol=1e-6 abstol={float(spec.get("abstol", 3e-15)):.3g} vntol=1e-9 gmin=1e-18 method=gear maxord=2',
             '.temp 27', f'VREF vref 0 {pc.VREF}', f'VCMV vcmv 0 {v_cm:.17g}', f'Vlow vlow 0 {vlow:.17g}',
             f'Vrect vrect 0 {float(spec["v_rect"]):.17g}', f'Vdref vdref 0 {v_ref:.17g}',
             f'Vped vped 0 {float(spec["norm_pedestal"]):.17g}', *pc.model_lines(model)]
    blocks = ['scorer', 'softmax', 'translinear', 'norm', 'switches']
    for b in blocks:
        lines.append(f'VDD_{b} vdd_{b} 0 {VDD}')
    count, columns = {}, []

    def mos(name, d, g, s_, b, kind, group, width=None, length=L, fam=None):
        if width is None:
            width = W_P if kind == 'pc' else W_N
        d, s_, b = (f'vdd_{group}' if n == 'vdd' else n for n in (d, s_, b))
        if fam is None and group != 'switches' and group != 'norm':
            fam = e2e.family(name, group)
        wf, lf = geometry.get(fam, (1., 1.)) if fam else (1., 1.)
        lines.append(f'{prefix}{name} {d} {g} {s_} {b} {kind} w={width * wf:.17g} l={length * lf:.17g}')
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

    # token DAC (duty-cycled) and crossbar
    seq = [ctx[j] for j in range(KEYS)] + [xq]
    for i in range(INPUTS):
        for sign, tag in ((1, 'p'), (-1, 'm')):
            pts = []
            for j, x in enumerate(seq):
                v = v_cm + sign * s * x[i]
                pts += [(j * T['t_write'] + (EDGE if j else 0), v), ((j + 1) * T['t_write'], v)]
            pts += [(T['t_query'] + T['t_write'] + EDGE, v_cm), (T['t_end'], v_cm)]
            lines.append(f'Vx{tag}{i} x{tag}{i} 0 {pwl(pts)}')
            columns += [(f'pwr_tok_i_{tag}{i}', f'i(Vx{tag}{i})'), (f'pwr_tok_v_{tag}{i}', f'v(x{tag}{i})')]
    cols = [(f'kc{d}', kgp[d], kgm[d]) for d in range(KDIMS)] + [(f'vc{e}', vgp[e], vgm[e]) for e in range(VDIMS)]
    for node, gp, gm in cols:
        for i in range(INPUTS):
            lines.append(f'Rx{node}p{i} xp{i} {node} {1 / gp[i]:.17g}')
            lines.append(f'Rx{node}m{i} xm{i} {node} {1 / gm[i]:.17g}')
    count['crossbar_resistors'] = 2 * INPUTS * len(cols)

    # KV cache, query hold and V_cm reference hold
    def hold(src, dst, clk, clkb):
        switch(f'sh_{dst}', src, dst, clk, clkb)
        lines.append(f'Chold_{dst} {dst} 0 {ch:.17g}')
        lines.append(f'.ic v({dst})={v_cm:.17g}')
        columns.append((f'hold_{dst}', f'v({dst})'))
    for j, (t0, t1) in enumerate(T['sample']):
        lines.append(f'Vphi{j} phi{j} 0 {pulse_high(t0, t1)}')
        lines.append(f'Vphib{j} phib{j} 0 {pulse_high(t0, t1, hi=0., lo=VDD)}')
        for d in range(KDIMS):
            hold(f'kc{d}', f'k{j}{d}', f'phi{j}', f'phib{j}')
        for e in range(VDIMS):
            hold(f'vc{e}', f'v{j}{e}', f'phi{j}', f'phib{j}')
    tq0, tq1 = T['query']
    lines.append(f'Vphiq phiq 0 {pulse_high(tq0, tq1)}')
    lines.append(f'Vphibq phibq 0 {pulse_high(tq0, tq1, hi=0., lo=VDD)}')
    for d in range(KDIMS):
        hold(f'kc{d}', f'qh{d}', 'phiq', 'phibq')
    hold('vcmv', 'vcmh', 'phiq', 'phibq')

    # anti-bump scorer, tail gating and capacitor integration
    lines.append(f'Iref_tail vdd_scorer tref {tail:.17g}')
    mos('tref', 'tref', 'tref', '0', '0', 'nc', 'scorer')
    lines.append('.nodeset v(tref)=.6')
    ta = max(T['t_int0'] - float(spec['t_warm']), EDGE)
    lines.append(f'Ven en 0 {pulse_high(ta, T["t_int1"] + EDGE)}')
    lines.append(f'Vrst rst 0 {pulse_high(0, T["t_int0"])}')
    lines.append(f'Vrstb rstb 0 {pulse_high(0, T["t_int0"], hi=0., lo=VDD)}')
    lines.append(f'Vint int 0 {pulse_high(T["t_int0"], T["t_int1"])}')
    lines.append(f'Vintb intb 0 {pulse_high(T["t_int0"], T["t_int1"], hi=0., lo=VDD)}')
    for j in range(KEYS):
        for d in range(KDIMS):
            t, x = f't{j}{d}', f'x{j}{d}'
            if model == 'ideal':
                lines.append(f'Sen_{j}{d} ts{j}{d} 0 en 0 swen')
            else:
                mos(f'en_{j}{d}', f'ts{j}{d}', 'en', '0', '0', 'nc', 'switches', W_SW, L_SW)
            mos(f'tail_{j}{d}', t, 'tref', f'ts{j}{d}', '0', 'nc', 'scorer')
            mos(f'oa_{j}{d}', f'sum{j}', f'qh{d}', t, '0', 'nc', 'scorer')
            mos(f'ob_{j}{d}', f'sum{j}', f'k{j}{d}', t, '0', 'nc', 'scorer')
            mos(f'ma_{j}{d}', x, f'qh{d}', t, '0', 'nc', 'scorer', width=S * W_N)
            mos(f'mb_{j}{d}', 'vdd', f'k{j}{d}', x, '0', 'nc', 'scorer', width=S * W_N)
            lines.append(f'.nodeset v({t})=.45 v({x})=.5 v(ts{j}{d})=.3')
        lines.append(f'Vsc_{j} psum{j} sum{j} 0')
        mos(f'pd_{j}', f'psum{j}', f'psum{j}', 'vdd', 'vdd', 'pc', 'scorer', width=W_N)
        mos(f'pc_{j}', f'pint{j}', f'psum{j}', 'vdd', 'vdd', 'pc', 'scorer', width=W_N)
        switch(f'int_{j}', f'pint{j}', f'gsc{j}', 'int', 'intb')
        switch(f'dump_{j}', f'pint{j}', 'vlow', 'intb', 'int')
        switch(f'rst_{j}', f'gsc{j}', 'vlow', 'rst', 'rstb')
        lines.append(f'.nodeset v(psum{j})=1.2 v(pint{j})={vlow:.17g}')
        lines.append(f'Cscore_{j} gsc{j} 0 {cs:.17g}')
        lines.append(f'.ic v(gsc{j})={vlow:.17g}')
        columns += [(f'gate_score_{j}', f'v(gsc{j})')]

    # Elfadel-Wyatt softmax into the translinear value weighting
    lines.append(f'Iref_ew ewr 0 {amp * IREF:.17g}')
    mos('ewr', 'ewr', 'ewr', 'vdd', 'vdd', 'pc', 'softmax')
    mos('ewt', 'ews', 'ewr', 'vdd', 'vdd', 'pc', 'softmax')
    lines.append('.nodeset v(ewr)=1.1 v(ews)=1.4')
    for j in range(KEYS):
        lines.append(f'Vew_{j} ewd{j} g_a{j} 0')
        mos(f'ew_{j}', f'ewd{j}', f'gsc{j}', 'ews', 'vdd', 'pc', 'softmax')
        columns.append((f'softmax_{j}', f'i(Vew_{j})'))
        mos(f'bd_{j}', f'g_a{j}', f'g_a{j}', '0', '0', 'nc', 'translinear')
        lines.append(f'.nodeset v(g_a{j})=.35 v(ewd{j})=.35')
        for e in range(VDIMS):
            tag = f'{j}_{e}'
            mos(f'bt_{tag}', f'bs_{tag}', f'g_a{j}', '0', '0', 'nc', 'translinear')
            mos(f'bp_{tag}', f'op{e}', f'v{j}{e}', f'bs_{tag}', '0', 'nc', 'translinear')
            mos(f'bm_{tag}', f'om{e}', 'vcmh', f'bs_{tag}', '0', 'nc', 'translinear')
            lines.append(f'.nodeset v(bs_{tag})=.5')

    # L1 layer norm: rectifier + translinear normalizer
    def nmos(name, d, g, s_):
        mos(name, d, g, s_, '0', 'nc', 'norm', w_ln, l_ln, fam='norm')

    def pmos(name, d, g, s_, b='vdd'):
        mos(name, d, g, s_, b, 'pc', 'norm', w_ln, l_ln, fam='norm')
    rb = {'vdd': 'vdd', 'vrect': 'vrect'}[spec.get('rect_bulk', 'vdd')]   # ideal law: no junctions, use the rail
    for e in range(VDIMS):
        lines.append(f'.nodeset v(op{e})=1.3 v(om{e})=1.3')
        lines.append(f'Vip{e} opd{e} op{e} 0')
        lines.append(f'Vim{e} omd{e} om{e} 0')
        pmos(f'lpd{e}', f'opd{e}', f'opd{e}', 'vdd')                # supplies I_p
        pmos(f'lpc{e}', f'r{e}', f'opd{e}', 'vdd')                  # +I_p into r_e
        pmos(f'lmd{e}', f'omd{e}', f'omd{e}', 'vdd')                # supplies I_m
        pmos(f'lmc{e}', f'mn{e}', f'omd{e}', 'vdd')
        nmos(f'lmn{e}', f'mn{e}', f'mn{e}', '0')
        nmos(f'lms{e}', f'r{e}', f'mn{e}', '0')                     # -I_m from r_e
        nmos(f'lrn{e}', f'r{e}', f'r{e}', 'vped')                   # D+ diode (log of D+ at gate r_e)
        pmos(f'lrp{e}', f'r{e}', f'r{e}', 'vrect', rb)              # D- diode from the low rail (bulk at VDD:
                                                                    # a rail-tied bulk forward-biases above the rail)
        pmos(f'lrc{e}', f'aq{e}', f'r{e}', 'vrect', rb)             # copy of D- (low rail)
        nmos(f'laq{e}', f'aq{e}', f'aq{e}', '0')                    # level shift: NMOS diode at ground,
        nmos(f'lar{e}', f'ap{e}', f'aq{e}', '0')                    # NMOS copy,
        pmos(f'lad{e}', f'ap{e}', f'ap{e}', 'vdd')                  # PMOS diode at VDD,
        pmos(f'las{e}', f'an{e}', f'ap{e}', 'vdd')                  # PMOS copy into
        nmos(f'lan{e}', f'an{e}', f'an{e}', 'vped')                 # the pedestaled log diode of D-
        lines.append(f'.nodeset v(opd{e})=1.3 v(omd{e})=1.3 v(r{e})=.5 v(mn{e})=.3 v(an{e})=.5 v(aq{e})=.3 v(ap{e})=1.3')
        columns += [(f'ip_{e}', f'i(Vip{e})'), (f'im_{e}', f'i(Vim{e})'), (f'rect_{e}', f'v(r{e})')]
    lines.append(f'Ieps vdd_norm aeps {i_eps:.17g}')
    nmos('leps', 'aeps', 'aeps', 'vped')
    lines.append(f'Iref_nt vdd_norm ntr {i_t:.17g}')
    nmos('lntr', 'ntr', 'ntr', '0')
    nmos('lnt', 'ns', 'ntr', '0')                                   # normalizer tail I_T
    lines.append('.nodeset v(ntr)=.4 v(ns)=.25 v(aeps)=.5')
    r_ti = v_u / i_t                          # transimpedance: V_U across the resistor at I_T
    lines.append(f'Vtr vtr 0 {pc.VREAD}')
    for e in range(VDIMS):                    # output current flows from the rail through R into the drain
        lines.append(f'Rnp{e} vtr npr{e} {r_ti:.17g}')
        lines.append(f'Vnp{e} npr{e} np{e} 0')
        lines.append(f'Rnm{e} vtr nmr{e} {r_ti:.17g}')
        lines.append(f'Vnm{e} nmr{e} nm{e} 0')
        lines.append(f'.nodeset v(np{e})=1.1 v(nm{e})=1.1')
        nmos(f'lop{e}', f'np{e}', f'r{e}', 'ns')
        nmos(f'lom{e}', f'nm{e}', f'an{e}', 'ns')
        columns += [(f'norm_p_{e}', f'i(Vnp{e})'), (f'norm_m_{e}', f'i(Vnm{e})')]
    nmos('loe', 'vdd_norm', 'aeps', 'ns')

    # ideal unity buffers (differential VCVS) from the transimpedance resistors into the DRN
    for e in range(VDIMS):     # V(nm) - V(np) = R (I_out+ - I_out-) = V_U n_e
        lines.append(f'Eu{e} du{e} vdref nm{e} np{e} 1')
        lines.append(f'Eum{e} dum{e} vdref np{e} nm{e} 1')
    lines.append(f'Vbp dbpos vdref {v_u:.17g}')
    lines.append(f'Vbm dbneg vdref {-v_u:.17g}')
    drn_in = ['du0', 'du1', 'dum0', 'dum1', 'dbpos', 'dbneg']
    H, C = g2.shape
    for h in range(H):
        for i, node in enumerate(drn_in):
            lines.append(f'Rg1_{i}_{h} {node} dh{h} {1 / g1[i, h]:.17g}')
        lines.append(f'Rsh_{h} dh{h} vdref {1 / g_s:.17g}')
        x = f'(V(vdref)-V(dh{h}))'
        lines.append(f'Bdi_{h} vdref dh{h} I = 0.1*({x} > 0.02 ? {x} : 1e-3*ln(1+exp({x}/1e-3)))')
        columns.append((f'drn_h_{h}', f'v(dh{h})'))
    for c in range(C):
        for h in range(H):
            lines.append(f'Rg2_{h}_{c} dh{h} do{c} {1 / g2[h, c]:.17g}')
        lines.append(f'Rso_{c} do{c} vdref {1 / g_s:.17g}')
        columns.append((f'drn_o_{c}', f'v(do{c})'))
    columns += [(f'drn_u_{i}', f'v({n})') for i, n in enumerate(drn_in)] + [('drn_ref', 'v(vdref)')]
    count['drn_resistors'] = H * 6 + H + C * H + C
    for b in blocks:
        columns.append((f'pwr_supply_{b}', f'i(VDD_{b})'))
    for ln in list(lines):
        if ln.startswith(('Vphi', 'Vrst', 'Vint', 'Ven ')):
            name, node = ln.split()[:2]
            columns += [(f'pwr_clk_i_{name}', f'i({name})'), (f'pwr_clk_v_{name}', f'v({node})')]
    tmax = min(T['t_end'] / 400, 2e-4)
    lines += ['.control', 'set noaskquit', 'set numdgt=15', 'set wr_vecnames', 'set wr_singlescale',
              f'tran {T["t_write"] / 200:.12g} {T["t_end"]:.12g} 0 {tmax:.12g}',
              'wrdata trace.txt ' + ' '.join(e_ for _, e_ in columns), 'quit', '.endc', '.end']
    return '\n'.join(lines) + '\n', columns, count


def decode(port, spec):
    amp_i = float(spec['amplitude']) * IREF
    i_t = float(spec['norm_tail'])
    out = {}
    out['o'] = np.array([(port[f'ip_{e}'][-1] - port[f'im_{e}'][-1]) for e in range(VDIMS)]) / amp_i
    # probe currents: I_p,e flows from the PMOS diode through Vip into op_e: i(Vip) > 0
    out['n'] = np.array([(port[f'norm_p_{e}'][-1] - port[f'norm_m_{e}'][-1]) for e in range(VDIMS)]) / i_t
    out['norm_sum'] = float(sum(port[f'norm_p_{e}'][-1] + port[f'norm_m_{e}'][-1] for e in range(VDIMS)) / i_t)
    ref = port['drn_ref'][-1]
    out['u'] = np.array([port[f'drn_u_{i}'][-1] - ref for i in range(6)])
    H = sum(1 for k in port if k.startswith('drn_h_'))
    C = sum(1 for k in port if k.startswith('drn_o_'))
    out['h'] = np.array([port[f'drn_h_{h}'][-1] - ref for h in range(H)])
    out['out'] = np.array([port[f'drn_o_{c}'][-1] - ref for c in range(C)])
    out['softmax'] = np.array([port[f'softmax_{j}'][-1] for j in range(KEYS)]) / amp_i
    t = port['time']
    i0 = np.searchsorted(t, t[-1] - .1 * (t[-1] - timing(spec)['t_int1']))
    out['late_drift_out'] = float(max(abs(port[f'drn_o_{c}'][-1] - port[f'drn_o_{c}'][i0]) for c in range(C)))
    out['energy'] = e2e.energy(port)
    g1, g2, g_s = np.asarray(spec['drn_g1']), np.asarray(spec['drn_g2']), float(spec['g_s'])
    p_drn = (g1 * (out['u'][:, None] - out['h'][None, :]) ** 2).sum() + g_s * (out['h'] ** 2).sum()
    p_drn += (g2 * (out['h'][:, None] - out['out'][None, :]) ** 2).sum() + g_s * (out['out'] ** 2).sum()
    out['drn_power_W'] = float(p_drn)
    return out


def simulate(spec, model, directory, executable, timeout=900):
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
    res['counts'], res['seconds'] = count, elapsed
    return res
