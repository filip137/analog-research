"""Explicit biased current-driven attention; ideal-law and generic BSIM3 devices.

Reference currents, voltage rails and readout load compliance are external ports.
Input sinks, attention/anchor biasing and readout mirrors are explicit MOS devices.
No query-key scorer or physical gradient multiplier is claimed by this model.
"""
from pathlib import Path
import subprocess
import time
import numpy as np

VDD = 1.8
VREF = .3
VREAD = 1.2
IREF = 1e-10
H0 = .01
OFFSET = 1.
UT = 8.617333262145e-5 * 300.15
LENGTH = 10e-6
WIDTH = 10e-6
CAP = 1e-12


def model_lines(model):
    if model == 'bsim3':
        return [
            '* Generic nominal BSIM3.3.0 cards; not a foundry PDK.',
            '.model nc nmos level=49 version=3.3.0 tox=10n nch=1e17 nsub=1e16 vth0=.7 u0=400 tnom=27',
            '.model pc pmos level=49 version=3.3.0 tox=10n nch=1e17 nsub=1e16 vth0=-.7 u0=150 tnom=27',
        ]
    if model != 'ideal':
        raise ValueError(model)
    return [
        '* Ideal symmetric weak-inversion laws, no body diodes or parasitics.',
        '.subckt nc d g s b params: w=10u l=10u',
        f'Bch d s I=1u*(w/l)*exp((V(g,b)-.7)/(1.5*{UT:.17g}))*(exp(-V(s,b)/{UT:.17g})-exp(-V(d,b)/{UT:.17g}))',
        '.ends nc',
        '.subckt pc d g s b params: w=10u l=10u',
        f'Bch s d I=1u*(w/l)*exp((V(b,g)-.7)/(1.5*{UT:.17g}))*(exp(V(s,b)/{UT:.17g})-exp(V(d,b)/{UT:.17g}))',
        '.ends pc',
    ]


def export(values, controls, model='bsim3', nudge=0., error=(.7,-.4), transient=False, initial=.3, transient_seconds=.02):
    values, controls = np.asarray(values), np.asarray(controls)
    if values.shape != (3,2) or controls.shape != (3,):
        raise ValueError('Expected three keys and two features')
    if np.min(OFFSET+values) <= 0 or np.min(controls) < 0:
        raise ValueError('Positive input currents and nonnegative controls required')
    lines = ['Biased three-key two-feature pseudoconductance attention',
             '.options reltol=1e-7 abstol=1e-20 vntol=1e-10 gmin=1e-18 method=gear maxord=2',
             '.temp 27', 'VDD vdd 0 1.8', f'VREF vref 0 {VREF}',
             *model_lines(model)]
    count = {'couplings':0,'anchors':0,'control_bias':0,'input_mirror':0,'read_mirror':0}
    def mos(name,d,g,s,b,kind,group):
        prefix='M' if model=='bsim3' else 'X'
        width=WIDTH/10 if kind=='pc' else WIDTH
        lines.append(f'{prefix}{name} {d} {g} {s} {b} {kind} w={width:.17g} l={LENGTH:.17g}')
        count[group]+=1
    for label,current in [('h',IREF),('h0',IREF*H0)]+[(f'a{j}',IREF*controls[j]) for j in range(3)]:
        lines.append(f'Iref_{label} vdd g_{label} {current:.17g}')
        mos('bias_'+label,'g_'+label,'g_'+label,'vref','0','nc','control_bias')
        lines.append(f'.nodeset v(g_{label})=.7')
    columns=[]
    for d in range(2):
        for j in range(4):
            tag=f'{j}_{d}'
            node=f'z{tag}'
            gate='g_h' if j<3 else 'g_h0'
            current=IREF*(OFFSET+values[j,d]) if j<3 else IREF*H0*OFFSET
            # Explicit two-NMOS input mirror, driven at its reference port.
            lines.append(f'Ivalue_{tag} vdd data_{tag} {current:.17g}')
            mos('iref_'+tag,'data_'+tag,'data_'+tag,'0','0','nc','input_mirror')
            lines.append(f'Vin_{tag} {node} sink_{tag} 0')
            mos('isink_'+tag,'sink_'+tag,'data_'+tag,'0','0','nc','input_mirror')
            # NMOS anchor supplied by a two-PMOS read mirror.
            mos('pr_'+tag,'sense_'+tag,'sense_'+tag,'vdd','vdd','pc','read_mirror')
            lines.append(f'Vanc_{tag} sense_{tag} anc_{tag} 0')
            mos('anchor_'+tag,'anc_'+tag,gate,node,'0','nc','anchors')
            mos('pcopy_'+tag,'read_'+tag,'sense_'+tag,'vdd','vdd','pc','read_mirror')
            lines.append(f'Vread_{tag} read_{tag} 0 {VREAD}')
            lines.append(f'Cstate_{tag} {node} 0 {CAP:.17g}')
            lines.append(f'.nodeset v({node})={initial} v(data_{tag})=.3 v(sense_{tag})=1.5')
            if transient:
                lines.append(f'.ic v({node})={initial}')
            columns += [(f'v_{tag}',f'v({node})'),(f'anchor_{tag}',f'i(Vanc_{tag})'),
                        (f'read_{tag}',f'i(Vread_{tag})'),(f'input_{tag}',f'i(Vin_{tag})'),
                        (f'sense_{tag}',f'v(sense_{tag})')]
        for j in range(3):
            lines.append(f'Vedge_{j}_{d} z{j}_{d} edge_{j}_{d} 0')
            mos(f'edge_{j}_{d}',f'edge_{j}_{d}',f'g_a{j}',f'z3_{d}','0','nc','couplings')
            columns += [(f'edge_{j}_{d}',f'i(Vedge_{j}_{d})')]
        lines.append(f'Inudge_{d} 0 z3_{d} {nudge*IREF*error[d]:.17g}')
    columns += [(f'gate_{k}',f'v(g_{k})') for k in ['h','h0','a0','a1','a2']]
    columns += [('supply','i(VDD)')]
    lines += ['.control','set noaskquit','set numdgt=15','set wr_vecnames','set wr_singlescale']
    # Without UIC, ngspice biases the other compact-model nodes consistently
    # with .ic on state nodes, then releases those initialization constraints.
    lines += [f'tran {transient_seconds/2000:.17g} {transient_seconds:.17g}' if transient else 'op']
    lines += ['wrdata trace.txt '+' '.join(expr for _,expr in columns),'quit','.endc','.end']
    return '\n'.join(lines)+'\n',columns,count


def simulate(values, controls, directory, executable, **kwargs):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    net,columns,count=export(values,controls,**kwargs)
    (directory/'circuit.cir').write_text(net)
    start=time.monotonic()
    proc=subprocess.run([str(executable),'-b','circuit.cir'],cwd=directory,
                        capture_output=True,text=True,timeout=10)
    elapsed=time.monotonic()-start
    (directory/'spice.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode or not (directory/'trace.txt').exists():
        raise RuntimeError(f'SPICE failed in {directory}: {(proc.stdout+proc.stderr)[-2000:]}')
    data=np.loadtxt(directory/'trace.txt',skiprows=1,ndmin=2)
    if data.shape[1] != len(columns)+1 or not np.isfinite(data).all():
        raise RuntimeError(f'Invalid trace shape/data {data.shape} in {directory}')
    if 'aborted' in (proc.stdout+proc.stderr).lower():
        raise RuntimeError(f'SPICE aborted an analysis despite zero exit status: {directory}')
    if kwargs.get('transient') and data[-1,0] < kwargs.get('transient_seconds',.02)*(1-1e-8):
        raise RuntimeError(f'Incomplete transient ending at {data[-1,0]} in {directory}')
    port={name:data[:,k+1] for k,(name,_) in enumerate(columns)}
    x=np.array([[port[f'read_{j}_{d}'][-1]/(IREF*(1 if j<3 else H0)) for d in range(2)] for j in range(4)])
    y=x[-1]-OFFSET
    voltage=np.array([[port[f'v_{j}_{d}'][-1] for d in range(2)] for j in range(4)])
    metrics={'min_node_voltage':float(voltage.min()),'max_node_voltage':float(voltage.max()),
             'min_anchor_vds':min(float(port[f'sense_{j}_{d}'][-1]-port[f'v_{j}_{d}'][-1]) for j in range(4) for d in range(2)),
             'mirror_ratio_error':max(abs(float(port[f'read_{j}_{d}'][-1]/port[f'anchor_{j}_{d}'][-1])-1) for j in range(4) for d in range(2)),
             'supply_power_W':float(-VDD*port['supply'][-1])}
    np.savez_compressed(directory/'trace.npz',time=data[:,0],**port)
    return {'y':y,'x_read':x,'voltage':voltage,'port':port,'counts':count,'metrics':metrics,'seconds':elapsed}


def reference(values,controls):
    w=np.asarray(controls)/(1+np.asarray(controls))
    return w@np.asarray(values)/(H0+w.sum())
