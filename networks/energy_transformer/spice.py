"""Explicit-element ngspice export, with a matched BJT softmax subcircuit.

Only the precision rectifier is a nonlinear behavioral voltage macro. The
matrix force is built from physical resistor port currents, CCCSs and VCCSs.
"""
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np
from networks.energy_transformer.circuit import checked_target

# ngspice 39 constants (src/include/ngspice/const.h), not rounded 26 mV.
BOLTZMANN = 1.38064852e-23
ELECTRON_CHARGE = 1.6021766208e-19
TEMPERATURE_C = 27.0
THERMAL_VOLTAGE = BOLTZMANN/ELECTRON_CHARGE*(TEMPERATURE_C+273.15)
TAIL_CURRENT = 1e-4
BJT_BETA = 1e8


def export_netlist(circuit, target, nudge=0.0, initial=None):
    p, c = circuit.parameters, circuit.components
    target = checked_target(p, target)
    if not np.isfinite(nudge) or p.stability_margin(nudge) <= 0:
        raise ValueError("Nudge outside the declared stable regime")
    initial = np.zeros(p.size) if initial is None else np.asarray(initial)
    p.split(initial)
    d, l, m = p.sizes
    lines = ["Signed reciprocal energy-attention circuit (ideal active macros)",
             "* Four explicit resistors per logical signed synapse; both ports sensed.",
             ".temp 27", ".options reltol=1e-9 abstol=1e-14 vntol=1e-10 gmin=1e-15 method=gear maxord=2",
             ".model qsoft NPN(IS=1e-15 BF=1e8 CJE=0 CJC=0 TF=0 TR=0)",
             "Vcc vcc 0 3", f"Itail emit 0 {TAIL_CURRENT:.17g}"]
    alpha = BJT_BETA/(BJT_BETA+1)
    for j in range(l):
        lines += [f"Ebase{j} base{j} 0 s{d+j} 0 {p.kappa*THERMAL_VOLTAGE:.17g}",
                  f"Vcol{j} vcc coll{j} 0", f"Qsoft{j} coll{j} base{j} emit qsoft",
                  f"Fread{j} 0 p{j} Vcol{j} 1",
                  f"Rread{j} p{j} 0 {1/(alpha*TAIL_CURRENT):.17g}"]
    for j in range(m):
        lines.append(f"Brect{j} r{j} 0 V=max(V(s{d+l+j}),0)")
    outputs = [f"s{i}" for i in range(d)]+[f"p{i}" for i in range(l)]+[f"r{i}" for i in range(m)]
    drives = np.concatenate((p.a, p.b, p.c))
    for i, activation in enumerate(outputs):
        lines += [f"Vdrive{i} drive{i} 0 {drives[i]:.17g}",
                  f"Rleak{i} s{i} drive{i} {1/c.conductance:.17g}",
                  f"Cstate{i} s{i} 0 {circuit.capacitances[i]:.17g} IC={initial[i]:.17g}",
                  f"Eplus{i} dp{i} 0 {activation} 0 1",
                  f"Eminus{i} dm{i} 0 {activation} 0 -1",
                  f"Vsplus{i} dp{i} rail{2*i} 0",
                  f"Vsminus{i} dm{i} rail{2*i+1} 0",
                  f"Fportp{i} s{i} 0 Vsplus{i} 0.5",
                  f"Fportm{i} 0 s{i} Vsminus{i} 0.5",
                  f"Gcomp{i} 0 s{i} {activation} 0 {c.conductance*circuit.compensation[i]:.17g}"]
    for index, (left, right, conductance) in enumerate(circuit.edges):
        lines.append(f"Rweight{index} rail{left} rail{right} {1/conductance:.17g}")
    for i, t in enumerate(target):
        lines += [f"Vtarget{i} target{i} 0 {t:.17g}",
                  f"Gnudge{i} s{i} 0 s{i} target{i} {c.conductance*nudge:.17g}"]
    vectors = ([f"v(s{i})" for i in range(p.size)]+[f"v(p{i})" for i in range(l)]
               +[f"v(r{i})" for i in range(m)]
               +[v for i in range(p.size) for v in (f"i(Vsplus{i})", f"i(Vsminus{i})")])
    step, stop = min(circuit.taus)/30, 80*max(circuit.taus)
    lines += [".control", "set noaskquit", "set numdgt=15", "set wr_singlescale",
              "set wr_vecnames", f"tran {step:.17g} {stop:.17g} 0 {step:.17g} uic",
              "wrdata waveform.tsv " + " ".join(vectors), "quit", ".endc", ".end"]
    return "\n".join(lines)+"\n"


@dataclass(frozen=True)
class SpiceTrace:
    time: np.ndarray
    states: np.ndarray
    probabilities: np.ndarray
    rectified: np.ndarray
    rail_currents: np.ndarray


def executable_identity(executable):
    path = shutil.which(str(executable))
    if path is None:
        raise FileNotFoundError(f"ngspice executable not found: {executable}")
    version = subprocess.run([path, "--version"], check=True, capture_output=True,
                             text=True, timeout=10)
    return Path(path).resolve(), version.stdout.strip()


def run_spice(circuit, target, directory, *, executable="ngspice", nudge=0.0,
              initial=None, timeout=30):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    netlist = directory/"circuit.cir"
    netlist.write_text(export_netlist(circuit, target, nudge, initial))
    path = shutil.which(str(executable))
    if path is None:
        raise FileNotFoundError(f"ngspice executable not found: {executable}")
    with (directory/"stdout.log").open("w") as log:
        subprocess.run([path, "-n", "-b", "-o", "spice.log", netlist.name],
                       cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)
    waveform = directory/"waveform.tsv"
    if not waveform.is_file():
        raise RuntimeError(f"ngspice did not produce a waveform: {directory}")
    data = np.loadtxt(waveform, skiprows=1, ndmin=2)
    p = circuit.parameters
    d, l, m = p.sizes
    expected = 1+3*p.size+l+m
    if data.shape[1] != expected or data.shape[0] < 100 or not np.isfinite(data).all():
        raise RuntimeError("Incomplete or nonfinite SPICE waveform")
    if np.any(np.diff(data[:, 0]) <= 0) or abs(data[-1, 0]-80*max(circuit.taus)) > 1e-10:
        raise RuntimeError("SPICE did not reach the requested terminal time")
    return SpiceTrace(data[:, 0], data[:, 1:1+p.size],
                      data[:, 1+p.size:1+p.size+l],
                      data[:, 1+p.size+l:1+p.size+l+m],
                      data[:, 1+p.size+l+m:])
