"""Native SPICE MOS devices for the staged attention implementation.

The matched LEVEL=1 triode cell has an analytically bilinear differential
output. BSIM3 is a generic compact-model sensitivity, not a foundry PDK.
Bias/readout sources remain ideal and are explicit in the netlist.
"""
import numpy as np

GATE_BIAS = 1.5
DRAIN_BIAS = .3
SIGNAL_SCALE = .1
THRESHOLD = .5
KP = 1e-4
PRODUCT_CURRENT = KP*SIGNAL_SCALE**2
INPUT_LIMIT = 4.
SOFTMAX_DEVICE = "mosfet_bsim3_regulated"
SOFTMAX_GATE_BIAS = .7
SOFTMAX_SCALE = .0345
SOFTMAX_TAIL = 1e-9
SOFTMAX_VDS = 3.


def softmax_model_lines():
    return [
        "* Generic BSIM3 characterization model; not a foundry PDK.",
        ".model attention_soft NMOS LEVEL=49 VERSION=3.3.0 TOX=10n NCH=1e17 NSUB=1e16 VTH0=0.5 U0=400 TNOM=27",
        f"Vmosbias mosbias 0 {SOFTMAX_GATE_BIAS}",
    ]


def softmax_bank(net, scores, kappa):
    """Matched current normalization at constant VDS and zero body bias.

    The high-impedance source-voltage sense controls a drain buffer; the buffer's
    return is vcc, not the floating source (which would short the tail KCL).
    At DC Id=f(Vgs) and sum Id=Itail. This is the gradient of a convex potential
    even when f is not exponential. Ideal regulation/isolation are staged macros.
    """
    source=net.name("mosbank_source");rail=net.name("mosbank_drain")
    net.lines += [f"{net.name('Itail')} {source} 0 {SOFTMAX_TAIL:.17g}",
                  f"{net.name('Eregulated')} {rail} vcc {source} 0 1"]
    outputs=[];gates=[];drains=[]
    for score in scores:
        gate=net.name("mosbank_gate");drain=net.name("mosbank_channel")
        out=net.name("probability");measure=net.name("Vbankmeasure")
        net.lines += [f"{net.name('Egate')} {gate} mosbias {score} 0 {kappa*SOFTMAX_SCALE:.17g}",
                      f"{measure} {rail} {drain} 0",
                      f"{net.name('Msoft')} {drain} {gate} {source} {source} attention_soft W=10u L=10u",
                      f"{net.name('Fprob')} 0 {out} {measure} {1e-5/SOFTMAX_TAIL:.17g}",
                      f"{net.name('Rprob')} {out} 0 100000"]
        outputs.append(out);gates.append(gate);drains.append(drain)
    net.banks.append({"source":source,"gates":gates,"drains":drains})
    return outputs


def model_lines():
    return [
        "* Nominal long-channel triode identity; not a process-specific model.",
        ".model attention_triode NMOS LEVEL=1 VTO=0.5 KP=100u LAMBDA=0 GAMMA=0 TNOM=27",
        f"Vproductgate productgate 0 {GATE_BIAS}",
        f"Vproductdrain productdrain 0 {DRAIN_BIAS}",
    ]


def product(net, left, right):
    """Four NMOS currents: ++ - -+ - +- + -- = KP*scale**2*x*y."""
    out=net.name("mosproduct")
    net.lines.append(f"{net.name('Rproduct')} {out} 0 {1/PRODUCT_CURRENT:.17g}")
    gates=[];drains=[]
    for sign in (1,-1):
        gate=net.name("productg");drain=net.name("productd")
        net.lines += [
            f"{net.name('Eproductg')} {gate} productgate {left} 0 {sign*SIGNAL_SCALE/2:.17g}",
            f"{net.name('Eproductd')} {drain} productdrain {right} 0 {sign*SIGNAL_SCALE/2:.17g}",
        ]
        gates.append(gate);drains.append(drain)
    for gi,gate in enumerate(gates):
        for di,rail in enumerate(drains):
            drain=net.name("productchannel");measure=net.name("Vproductmeasure")
            sign=1 if gi==di else -1
            net.lines += [f"{measure} {rail} {drain} 0",
                          f"{net.name('Mproduct')} {drain} {gate} 0 0 attention_triode W=10u L=10u",
                          f"{net.name('Fproduct')} 0 {out} {measure} {sign}"]
    net.products.append({"left":left,"right":right,"output":out})
    return out


def domain_metrics(left,right):
    left,right=np.asarray(left),np.asarray(right)
    min_vds=DRAIN_BIAS-SIGNAL_SCALE*np.max(np.abs(right))/2
    max_vds=DRAIN_BIAS+SIGNAL_SCALE*np.max(np.abs(right))/2
    min_overdrive=GATE_BIAS-THRESHOLD-SIGNAL_SCALE*np.max(np.abs(left))/2
    return {"minimum_vds":float(min_vds),"minimum_triode_margin":float(min_overdrive-max_vds),
            "maximum_operand":float(max(np.max(np.abs(left)),np.max(np.abs(right))))}
