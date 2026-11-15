"""Render the circuit connection diagram and plots from explicit native bundles."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

INK = '#19354a'
BLUE = '#227bb6'
ORANGE = '#cf6832'


def diagram(destination):
    fig = plt.figure(figsize=(15, 12), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[.9, 1.2], width_ratios=[1.7, 1])
    top = fig.add_subplot(grid[0, :])
    cell = fig.add_subplot(grid[1, 0])
    bias = fig.add_subplot(grid[1, 1])
    for ax in (top, cell, bias):
        ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis('off')

    def wire(ax, points, color=INK):
        ax.plot(*zip(*points), color=color, lw=1.5, solid_capstyle='round')

    def label(ax, x, y, text, **kwargs):
        ax.text(x, y, text, ha=kwargs.pop('ha', 'center'), va='center',
                fontsize=kwargs.pop('fontsize', 10), color=INK, **kwargs)

    def node(ax, x, y):
        ax.add_patch(Circle((x, y), .055, color=INK))

    def mos(ax, x, y, name, kind='N', w=1.5, h=1.2):
        # Terminal-labelled blocks are intentional: this is a connection diagram.
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h,
                              facecolor='#eef5f9', edgecolor=INK, lw=1.2))
        label(ax, x, y, name, fontsize=9)
        label(ax, x, y+h*.37, 'D' if kind=='N' else 'S', fontsize=8)
        label(ax, x, y-h*.37, 'S' if kind=='N' else 'D', fontsize=8)
        label(ax, x-w*.38, y, 'G', fontsize=8)
        return (x, y+h/2), (x, y-h/2), (x-w/2, y)

    top.set_title('A. One feature channel: three value states couple to one output state',
                  loc='left', fontsize=15, color=INK, pad=14)
    for j, x in enumerate([2.8, 6.8, 10.8]):
        top.add_patch(Rectangle((x-1, 7.3), 2, 1.4, facecolor='#f8fafb', edgecolor=INK))
        label(top, x, 8, f'State cell {j},d\n(details below)')
        wire(top, [(x, 7.3), (x, 6.5)]); node(top, x, 6.5)
        label(top, x+.2, 6.7, f'z{j},d', ha='left')
        d, s, g = mos(top, x, 4.7, f'Medge {j},d', w=1.6, h=2)
        wire(top, [(x, 6.5), d]); wire(top, [s, (x, 2.5)])
        wire(top, [(x-1.6, 4.7), g], BLUE)
        label(top, x-1.6, 5.25, f'g_a{j}', fontsize=10)
    wire(top, [(2.8, 2.5), (14.4, 2.5)])
    node(top, 14.4, 2.5)
    top.add_patch(Rectangle((13.25, 6), 2.3, 1.6, facecolor='#f8fafb', edgecolor=INK))
    label(top, 14.4, 6.8, 'Output cell 3,d\nh = 0.01')
    wire(top, [(14.4, 6), (14.4, 2.5)])
    label(top, 14.4, 1.8, 'z3,d')
    label(top, 8, .7, 'Repeat for d = 0, 1. Gates g_a0, g_a1, g_a2 are shared across features.\nAll coupling bulks = 0 V. D and S are free state terminals; current may flow either way.', fontsize=11)

    cell.set_title('B. Each of the eight state cells: 5 MOS + 1 capacitor',
                   loc='left', fontsize=13, color=INK, pad=15)
    wire(cell, [(1, 9.3), (14.9, 9.3)])
    label(cell, 1, 9.65, 'VDD = 1.8 V', ha='left')
    # Reference PMOS and its readout copy.
    ps, pd, pg = mos(cell, 4.5, 8.1, 'Pr', 'P')
    rs, rd, rg = mos(cell, 11, 8.1, 'Pcopy', 'P')
    wire(cell, [ps, (4.5, 9.3)]); wire(cell, [rs, (11, 9.3)])
    wire(cell, [pd, (4.5, 6.8)]); node(cell, 4.5, 6.8)
    wire(cell, [(4.5, 6.8), (3.1, 6.8), (3.1, 8.1), pg])
    wire(cell, [(4.5, 6.8), (9.5, 6.8), (9.5, 8.1), rg])
    label(cell, 6.7, 7.2, 'sense_i = mirror gate')
    wire(cell, [rd, (11, 5.9)])
    label(cell, 12, 6.1, 'Read current R_i\nload held at 1.2 V', ha='left', fontsize=9)
    ad, ass, ag = mos(cell, 4.5, 5.6, 'Anchor')
    wire(cell, [(4.5, 6.8), ad]); wire(cell, [ass, (4.5, 4)])
    wire(cell, [(1.8, 5.6), ag], BLUE)
    label(cell, 1.7, 6.2, 'g_h or g_h0')
    wire(cell, [(4.5, 4), (13.7, 4)]); node(cell, 4.5, 4)
    label(cell, 4, 4, 'z_i', ha='right', fontsize=12)
    label(cell, 12.8, 4.45, 'To coupling MOS')
    # Input mirror: diode-connected reference and copying current sink.
    nd, ns, ng = mos(cell, 2, 2.1, 'Nref')
    sd, ss, sg = mos(cell, 6.8, 2.1, 'Nsink')
    wire(cell, [(2, 3.55), nd]); node(cell, 2, 3.1)
    wire(cell, [(2, 3.1), (.65, 3.1), (.65, 2.1), ng])
    wire(cell, [(2, 3.1), (5.5, 3.1), (5.5, 2.1), sg])
    label(cell, 1.65, 4.35, 'External J_i', fontsize=9)
    cell.annotate('', xy=(2, 3.5), xytext=(2, 4.05), arrowprops={'arrowstyle':'->', 'color':INK})
    wire(cell, [(6.8, 4), sd]); node(cell, 6.8, 4)
    wire(cell, [ns, (2, .9)]); wire(cell, [ss, (6.8, .9)])
    wire(cell, [(2, .9), (13, .9)])
    wire(cell, [(10.7, 4), (10.7, 2.6)])
    wire(cell, [(10.1, 2.6), (11.3, 2.6)])
    wire(cell, [(10.1, 2.35), (11.3, 2.35)])
    wire(cell, [(10.7, 2.35), (10.7, .9)])
    label(cell, 12, 2.5, 'C = 1 pF')
    label(cell, 13.3, .9, '0 V', ha='left')
    label(cell, 8, .15, 'NMOS bulks = 0 V; PMOS bulks = VDD. The output has an added nudge-current port.', fontsize=9)

    bias.set_title('C. Five shared current-to-gate controllers',
                   loc='left', fontsize=12, color=INK, pad=15)
    label(bias, 7, 9.35, 'External reference current I_c')
    bias.annotate('', xy=(7, 8.1), xytext=(7, 8.9), arrowprops={'arrowstyle':'->', 'color':INK})
    bd, bs, bg = mos(bias, 7, 6.4, 'Nbias', w=3, h=2)
    wire(bias, [(7, 8.1), bd]); node(bias, 7, 7.8)
    wire(bias, [(7, 7.8), (3.7, 7.8), (3.7, 6.4), bg])
    wire(bias, [(7, 7.8), (14.4, 7.8)], BLUE)
    label(bias, 11.5, 8.4, 'Gate bias g_c', fontsize=11)
    wire(bias, [bs, (7, 4.8)])
    label(bias, 7, 4.35, 'Source = VREF = 0.3 V\nBulk = 0 V')
    label(bias, 8, 2.6, 'c = h:      I_c = 100 pA\nc = h0:    I_c = 1 pA\nc = aj:     I_c = 100 pA × a_j', fontsize=11)
    label(bias, 8, .8, 'NMOS W/L = 10 µm / 10 µm\nPMOS W/L = 1 µm / 10 µm\nGeneric BSIM3; 27 °C', fontsize=11)
    fig.suptitle('Pseudoconductance attention — exact connection pattern', fontsize=19, color=INK)
    for ext in ('svg', 'png'):
        fig.savefig(destination/f'circuit.{ext}', dpi=160, facecolor='white')
    plt.close(fig)


def evidence(bundle, destination):
    art = bundle/'artifacts'
    summary = json.loads((art/'summary.json').read_text())
    routing = np.load(art/'routing.npz')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
    target = routing['targets'].ravel()
    ax[0].scatter(target, routing['uniform'].ravel(), s=22, color='#aaaaaa', alpha=.65, label='Uniform controls')
    ax[0].scatter(target, routing['selective'].ravel(), s=18, color=BLUE, label='Selective controls')
    ax[0].plot([-.8, .8], [-.8, .8], color=INK, lw=1, ls='--')
    ax[0].set(xlabel='Requested value component', ylabel='Circuit output', title='36 fresh arrays × 2 features', xlim=(-.8,.8), ylim=(-.8,.8))
    ax[0].legend(fontsize=9)
    mix = summary['mixtures']
    t = [m['t'] for m in mix]
    for d, color in enumerate([BLUE, ORANGE]):
        ax[1].plot(t, [m['reference'][d] for m in mix], color=color, label=f'Feature {d}: ideal operator')
        ax[1].plot(t, [m['output'][d] for m in mix], 'o', ms=4, color=color)
    ax[1].set(xlabel='t: a = (5t, 5(1−t), 0.001)', ylabel='Circuit output', title='Controlled mixing (dots: BSIM3)')
    ax[1].legend(fontsize=8)
    settlement=[]
    for j, color in enumerate([BLUE, ORANGE]):
        d=np.load(art/f'transient-{j}'/'trace.npz')
        y=np.stack([d[f'read_3_{k}']/1e-12-1 for k in range(2)],axis=1)
        err=np.max(abs(y-np.array(summary['example']['output'])),axis=1)
        tail=np.maximum.accumulate(err[::-1])[::-1]
        settled=np.flatnonzero(tail<1e-3)
        settlement.append(float(d['time'][settled[0]]) if len(settled) else None)
        ax[2].semilogy(d['time']*1000, np.maximum(err,1e-13), color=color, label=f'Initial state {(.2,.4)[j]} V')
    ax[2].axhline(1e-3, color=INK, ls='--', lw=1, label='Declared tolerance')
    ax[2].set(xlabel='Time (ms)', ylabel='Maximum output error vs. DC', title='Slow bias settling', xlim=(0,100), ylim=(1e-6,10))
    ax[2].legend(fontsize=8)
    for a in ax:
        a.spines[['top','right']].set_visible(False); a.grid(alpha=.15)
    fig.suptitle('Generic BSIM3 simulation · supplied attention controls · 51 MOS devices', fontsize=14)
    for ext in ('svg','png'):
        fig.savefig(destination/f'validation.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    return settlement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    diagram(args.output)
    print(json.dumps({'settled_within_1e-3_seconds':evidence(args.bundle,args.output)}))


if __name__=='__main__':
    main()
