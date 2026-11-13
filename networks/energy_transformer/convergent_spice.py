"""Explicit analog-block netlist for the convergent ET prototype.

State RCs and differential weight resistors are explicit. Selectable attention
uses native MOS product cells and a regulated BSIM3 current bank, or ideal
reference blocks. Normalization, rectification and buffers remain ideal.
No behavioral source implements the whole vector field or matrix product.
"""
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import numpy as np

from networks.energy_transformer.convergent import Parameters, Boundary, ports
from networks.energy_transformer.spice import THERMAL_VOLTAGE, TAIL_CURRENT, BJT_BETA
from networks.energy_transformer import mosfet, transistor

G0 = 1e-4
TIME_UNIT = 1e-3
TAUS = (1.,1.3)


class Builder:
    def __init__(self, softmax_device=mosfet.DEVICE, multiplier="ideal"):
        if softmax_device not in ("bjt",mosfet.DEVICE,transistor.SOFTMAX_DEVICE):
            raise ValueError("Unknown softmax device")
        self.softmax_device = softmax_device
        if multiplier not in ("ideal","mos_triode"):
            raise ValueError("Unknown multiplier")
        self.multiplier=multiplier
        self.lines = ["Convergent energy transformer -- staged analog circuit",
            "* Quasistatic multiplier/normalizer/rectifier macros; finite bandwidth untested.",
            f"* softmax_device={softmax_device}", ".temp 27", "Vcc vcc 0 3"]
        if softmax_device == transistor.SOFTMAX_DEVICE:
            self.lines += [".options reltol=1e-6 abstol=1e-18 vntol=1e-9 gmin=1e-18 method=gear maxord=2"]+transistor.softmax_model_lines()
        elif softmax_device == mosfet.DEVICE:
            self.lines += [".options reltol=1e-9 abstol=1e-18 vntol=1e-10 gmin=1e-18 method=gear maxord=2",
                           f"Vmosbias mosbias 0 {mosfet.GATE_BIAS:.17g}"]+mosfet.subcircuit()
        else:
            self.lines += [".options reltol=1e-9 abstol=1e-14 vntol=1e-10 gmin=1e-15 method=gear maxord=2",
                           ".model qsoft NPN(IS=1e-15 BF=1e8 CJE=0 CJC=0 TF=0 TR=0)"]
        self.serial = 0
        self.products=[]
        if multiplier=="mos_triode": self.lines += transistor.model_lines()
        self.negatives = {}
        self.positives = {}
        self.banks = []

    def name(self,prefix):
        self.serial += 1
        return f"{prefix}{self.serial}"

    def negative(self,node):
        if node not in self.negatives:
            neg = self.name("negative")
            self.lines.append(f"E{neg} {neg} 0 {node} 0 -1")
            self.negatives[node] = neg
        return self.negatives[node]

    def positive(self,node):
        # State capacitors must not directly supply the projection's resistor load.
        if node not in self.positives:
            pos = self.name("positive")
            self.lines.append(f"E{pos} {pos} 0 {node} 0 1")
            self.positives[node] = pos
        return self.positives[node]

    def linear(self,matrix,inputs,prefix):
        """Each signed coefficient is two positive resistors on +/- signal rails."""
        outputs = []
        for row in matrix:
            out = self.name(prefix)
            sense = self.name("sense")
            measurement = self.name("Vmeasure")
            self.lines += [f"{measurement} {sense} 0 0", f"{self.name('Fread')} 0 {out} {measurement} 1", f"{self.name('Rread')} {out} 0 {1/G0:.17g}"]
            for weight,node in zip(row,inputs):
                gp,gm = G0*(.02+max(weight,0.)), G0*(.02+max(-weight,0.))
                self.lines += [f"{self.name('Rweight')} {self.positive(node)} {sense} {1/gp:.17g}",
                               f"{self.name('Rweight')} {self.negative(node)} {sense} {1/gm:.17g}"]
            outputs.append(out)
        return outputs

    def product(self,left,right):
        if self.multiplier=="mos_triode": return transistor.product(self,left,right)
        out = self.name("product")
        self.lines.append(f"{self.name('Bmultiply')} {out} 0 V=V({left})*V({right})")
        return out

    def summation(self,nodes,prefix="sum"):
        out = self.name(prefix)
        self.lines.append(f"{self.name('Rsum')} {out} 0 {1/G0:.17g}")
        for node in nodes:
            self.lines.append(f"{self.name('Gsum')} 0 {out} {node} 0 {G0:.17g}")
        return out

    def softmax(self,scores,kappa):
        if self.softmax_device == transistor.SOFTMAX_DEVICE:
            return transistor.softmax_bank(self,scores,kappa)
        emitter = self.name("emitter")
        tail = mosfet.TAIL_CURRENT if self.softmax_device == mosfet.DEVICE else TAIL_CURRENT
        self.lines.append(f"{self.name('Itail')} {emitter} 0 {tail:.17g}")
        outputs=[]
        gates=[]
        drains=[]
        alpha=BJT_BETA/(BJT_BETA+1)
        for score in scores:
            base,collector,out = self.name("base"),self.name("collector"),self.name("probability")
            measure = self.name("Vcollector")
            if self.softmax_device == mosfet.DEVICE:
                self.lines += [f"{self.name('Egate')} {base} mosbias {score} 0 {kappa*mosfet.SLOPE_FACTOR*mosfet.THERMAL_VOLTAGE:.17g}",
                               f"{measure} vcc {collector} 0", f"{self.name('Xsoftmos')} {collector} {base} {emitter} mosweak",
                               f"{self.name('Fprob')} 0 {out} {measure} {TAIL_CURRENT/tail:.17g}",
                               f"{self.name('Rprob')} {out} 0 {1/TAIL_CURRENT:.17g}"]
            else:
                self.lines += [f"{self.name('Ebase')} {base} 0 {score} 0 {kappa*THERMAL_VOLTAGE:.17g}",
                               f"{measure} vcc {collector} 0", f"{self.name('Qsoft')} {collector} {base} {emitter} qsoft",
                               f"{self.name('Fprob')} 0 {out} {measure} 1",f"{self.name('Rprob')} {out} 0 {1/(alpha*TAIL_CURRENT):.17g}"]
            outputs.append(out)
            gates.append(base)
            drains.append(collector)
        self.banks.append({"source":emitter,"gates":gates,"drains":drains})
        return outputs


def export_netlist(parameters, boundary, nudge=0., initial=None, *, softmax_device=mosfet.DEVICE, multiplier="ideal", terminal_only=False, duration=80.):
    p,b = parameters,boundary.checked(parameters,nudge)
    n,d = p.shape
    h,a,_ = p.query.shape
    patch = p.output_shape[1]
    initial = np.zeros(p.size) if initial is None else initial
    z0,y0 = p.split(initial)
    net = Builder(softmax_device,multiplier)
    z = np.array([[f"z_{i}_{j}" for j in range(d)] for i in range(n)])
    y = np.array([[f"y_{i}_{j}" for j in range(patch)] for i in range(n)])
    g=[]; enc=[]; pos=[]
    for i in range(n):
        xs=[]; bs=[]
        for j in range(patch):
            xnode=f"x_{i}_{j}"; tnode=f"target_{i}_{j}"
            net.lines += [f"V{xnode} {xnode} 0 {b.inputs[i,j]:.17g}", f"V{tnode} {tnode} 0 {b.target[i,j]:.17g}"]
            net.lines += [f"Ryleak_{i}_{j} {y[i,j]} 0 {1/G0:.17g}",
                          f"Cstatey_{i}_{j} {y[i,j]} 0 {G0*TIME_UNIT*TAUS[1]:.17g} IC={y0[i,j]:.17g}",
                          f"Gnudge_{i}_{j} {y[i,j]} 0 {y[i,j]} {tnode} {G0*nudge*b.mask[i,j]:.17g}"]
            xs.append(xnode)
        for j in range(d):
            bias=f"bias_{i}_{j}"
            net.lines += [f"V{bias} {bias} 0 {p.position[i,j]:.17g}",
                          f"Rzleak_{i}_{j} {z[i,j]} 0 {1/G0:.17g}",
                          f"Cstatez_{i}_{j} {z[i,j]} 0 {G0*TIME_UNIT*TAUS[0]:.17g} IC={z0[i,j]:.17g}"]
            bs.append(bias)
        enc.append(net.linear(p.encoder,xs,"encode")); pos.append(bs)
        mean=f"mean_{i}"; variance=f"variance_{i}"; scale=f"scale_{i}"
        meanexpr="+".join(f"V({node})" for node in z[i])
        varexpr="+".join(f"(V({node})-V({mean}))^2" for node in z[i])
        net.lines += [f"Bmean_{i} {mean} 0 V=({meanexpr})/{d}",
                      f"Bvariance_{i} {variance} 0 V=({varexpr})/{d}+{p.epsilon:.17g}",
                      f"Bsqrt_{i} {scale} 0 V=sqrt(V({variance}))"]
        gi=[]
        for j in range(d):
            out=f"g_{i}_{j}"
            net.lines.append(f"Bnormalize_{i}_{j} {out} 0 V=(V({z[i,j]})-V({mean}))/V({scale})")
            gi.append(out)
        g.append(gi)
    g=np.array(g)
    q=np.array([[net.linear(p.query[t],g[i],"query") for i in range(n)] for t in range(h)])
    k=np.array([[net.linear(p.key[t],g[i],"key") for i in range(n)] for t in range(h)])
    probabilities=[]
    for t in range(h):
        rows=[]
        for i in range(n):
            scores=[net.summation([net.product(q[t,i,c],k[t,j,c]) for c in range(a)],"score") for j in range(n)]
            rows.append(net.softmax(scores,p.kappa))
        probabilities.append(rows)
    probabilities=np.array(probabilities)
    rectified=[]; drivez=[]; drivey=[]
    for i in range(n):
        hidden=net.linear(p.memory,g[i],"memory")
        relu=[]
        for node in hidden:
            out=net.name("relu")
            net.lines.append(f"{net.name('Brectify')} {out} 0 V=max(V({node}),0)")
            relu.append(out)
        rectified.append(relu)
        returns=[net.linear(p.memory.T,relu,"memoryreturn"),net.linear(p.decoder.T,y[i],"decoderreturn")]
        for t in range(h):
            qr=[net.summation([net.product(probabilities[t,i,j],k[t,j,c]) for j in range(n)],"queryfeedback") for c in range(a)]
            kr=[net.summation([net.product(probabilities[t,j,i],q[t,j,c]) for j in range(n)],"keyfeedback") for c in range(a)]
            returns += [net.linear(p.query[t].T,qr,"queryreturn"),net.linear(p.key[t].T,kr,"keyreturn")]
        dz=[net.summation([enc[i][j],pos[i][j]]+[r[j] for r in returns],"tokendrive") for j in range(d)]
        dy=net.linear(p.decoder,g[i],"outputdrive")
        for node,drive in zip(z[i],dz): net.lines.append(f"{net.name('Gtoken')} 0 {node} {drive} 0 {G0:.17g}")
        for node,drive in zip(y[i],dy): net.lines.append(f"{net.name('Goutput')} 0 {node} {drive} 0 {G0:.17g}")
        drivez.append(dz); drivey.append(dy)
    groups={"z":z,"y":y,"g":g,"q":q,"k":k,"p":probabilities,"r":np.array(rectified),"token_drive":np.array(drivez),"output_drive":np.array(drivey)}
    if softmax_device in (mosfet.DEVICE,transistor.SOFTMAX_DEVICE):
        groups["softmax_sources"]=np.array([bank["source"] for bank in net.banks]).reshape(h,n)
        groups["softmax_gates"]=np.array([bank["gates"] for bank in net.banks]).reshape(h,n,n)
        groups["softmax_drains"]=np.array([bank["drains"] for bank in net.banks]).reshape(h,n,n)
    if multiplier=="mos_triode":
        for key in ("left","right","output"):
            groups["product_"+key]=np.array([cell[key] for cell in net.products])
    if softmax_device == mosfet.DEVICE:
        # The ideal exponential macro lacks a compact model's Newton limiting.
        # Seed algebraic voltages consistently with the prescribed RC initial
        # states; this changes no component law or physical initial state.
        initial_ports=ports(p,np.asarray(initial))
        initial_values={"vcc":mosfet.DRAIN_VOLTAGE,"mosbias":mosfet.GATE_BIAS}
        for i in range(n):
            initial_values[f"mean_{i}"]=z0[i].mean()
            initial_values[f"variance_{i}"]=initial_ports["s"][i,0]**2
            initial_values[f"scale_{i}"]=initial_ports["s"][i,0]
        for key in ("g","q","k","p","r"):
            initial_values.update(zip(groups[key].ravel(),initial_ports[key].ravel()))
        for index,bank in enumerate(net.banks):
            head,row=divmod(index,n)
            scores=initial_ports["k"][head]@initial_ports["q"][head,row]
            gates=mosfet.GATE_BIAS+p.kappa*mosfet.SLOPE_FACTOR*mosfet.THERMAL_VOLTAGE*scores
            amplitude=np.exp((gates-mosfet.THRESHOLD)/(mosfet.SLOPE_FACTOR*mosfet.THERMAL_VOLTAGE)).sum()
            source=-mosfet.THERMAL_VOLTAGE*np.log(mosfet.TAIL_CURRENT/(mosfet.CURRENT_SCALE*amplitude)+np.exp(-mosfet.DRAIN_VOLTAGE/mosfet.THERMAL_VOLTAGE))
            initial_values[bank["source"]]=source
            initial_values.update(zip(bank["gates"],gates))
            initial_values.update({node:mosfet.DRAIN_VOLTAGE for node in bank["drains"]})
        net.lines += [f".ic V({node})={float(value):.17g}" for node,value in initial_values.items()]
    if softmax_device == transistor.SOFTMAX_DEVICE:
        # Constrain only RC states during the bias solve; SPICE initializes all
        # compact-model charges consistently before releasing these constraints.
        for node,value in zip(np.r_[z.ravel(),y.ravel()],initial):
            net.lines.append(f".ic V({node})={float(value):.17g}")
    vectors=[f"v({node})" for nodes in groups.values() for node in nodes.ravel()]
    if not np.isfinite(duration) or duration<=0: raise ValueError("Expected positive duration")
    step,stop=TIME_UNIT*(1. if terminal_only else .05),duration*TIME_UNIT
    net.lines += [".control","set noaskquit","set numdgt=15","set wr_singlescale","set wr_vecnames",
                  f"tran {step:.17g} {stop:.17g} 0 {step:.17g}"+("" if softmax_device==transistor.SOFTMAX_DEVICE else " uic")]
    if terminal_only:
        net.lines += ["let finaltime = time[length(time)-1]"]
        for i,vector in enumerate(vectors): net.lines.append(f"let endpoint{i} = {vector}[length(time)-1]")
        net.lines += ["setscale finaltime"]
        net.lines += ["wrdata waveform.tsv finaltime "+" ".join(f"endpoint{i}" for i in range(len(vectors)))]
    else:
        net.lines += ["wrdata waveform.tsv "+" ".join(vectors)]
    net.lines += ["quit",".endc",".end"]
    return "\n".join(net.lines)+"\n",{name:nodes.shape for name,nodes in groups.items()}


@dataclass(frozen=True)
class SpiceTrace:
    time: np.ndarray
    states: np.ndarray
    ports: dict


def run_spice(parameters,boundary,directory,*,executable="ngspice",nudge=0.,initial=None,timeout=30,softmax_device=mosfet.DEVICE,multiplier="ideal",terminal_only=False,duration=80.):
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    text,shapes=export_netlist(parameters,boundary,nudge,initial,softmax_device=softmax_device,multiplier=multiplier,terminal_only=terminal_only,duration=duration)
    (directory/"circuit.cir").write_text(text)
    executable=shutil.which(str(executable))
    if executable is None: raise FileNotFoundError("ngspice executable not found")
    with (directory/"stdout.log").open("w") as log:
        subprocess.run([executable,"-n","-b","-o","spice.log","circuit.cir"],cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
    waveform=directory/"waveform.tsv"
    if not waveform.is_file(): raise RuntimeError(f"SPICE did not produce waveform: {directory}")
    data=np.loadtxt(waveform,skiprows=1,ndmin=2)
    # Scalar vectors carry a dummy SPICE scale column before explicit finaltime.
    if terminal_only: data=data[:,1:]
    expected=1+sum(np.prod(shape) for shape in shapes.values())
    if data.shape[1]!=expected or data.shape[0]<(1 if terminal_only else 100) or not np.isfinite(data).all():
        raise RuntimeError("Incomplete or nonfinite SPICE waveform")
    if abs(data[-1,0]-duration*TIME_UNIT)>1e-10 or np.any(np.diff(data[:,0])<=0):
        raise RuntimeError("Incomplete SPICE time coverage")
    values={};offset=1
    for name,shape in shapes.items():
        width=np.prod(shape)
        values[name]=data[:,offset:offset+width].reshape((len(data),)+shape)
        offset+=width
    states=np.concatenate((values["z"].reshape(len(data),-1),values["y"].reshape(len(data),-1)),axis=1)
    return SpiceTrace(data[:,0]/TIME_UNIT,states,values)
