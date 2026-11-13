# Ideal analog energy-attention circuit

This prototype extends the analog computation to reciprocal softmax retrieval
and an optional tied ReLU branch. It is a separate architecture from the
repository's perfect-diode DRNs and from ordinary untied transformer attention.

The [scientific circuit design](https://github.com/filip137/on-chip-learning/blob/main/campaigns/analog-energy-transformer/circuit-design.md)
derives the four-resistor signed synapse, current-sensing feedback, loading
compensation, KCL, Lyapunov energy and sufficient stability bound. The shared
record is initially local until published by the study owner.

`circuit.py` provides parameters, explicit resistor port currents, RC
integration and independent energy/root references. `spice.py` exports and
runs ngspice netlists with actual resistors, capacitors and a matched NPN
softmax pool. Ideal controlled sources implement buffers, current conveyors
and compensation; the precision ReLU is a behavioral voltage macro. Thermal
constants are matched to ngspice 39. A different simulator/version must pass
the softmax and circuit checks before its output is used.

`methods/energy_transformer.py` computes EqProp from phase correlations. It
also provides implicit and loss-finite-difference references, and an ideal
software update controller that regenerates both conductances and compensation.
No trajectory is differentiated to obtain the EqProp learning signal. Physical
sample-and-hold, analog multipliers, device programming, noise and bandwidth
remain unimplemented.

Run the bounded characterization through the public interface:

```bash
python -m ebl characterize \
  --config examples/analog_energy_transformer_circuit.json \
  --output-dir /absolute/path/to/results
python -m ebl runs inspect /absolute/path/to/results/RUN_ID \
  --verify-artifacts --require-complete
```

Set `spice_executable` in a copy of the config to an installed ngspice path.
The committed example names the campaign's locally built executable. The
runner creates one native bundle per case, checks all declared thresholds,
and returns nonzero if any scientific check fails. A complete bundle and a
passing scientific result are distinct; both appear in its summary.

The three example cases verify signed couplings, mixed/all-active ReLUs and
softmax alone. They include three initial states, free and nudged equilibria,
a beta sweep, all five parameter/input gradient groups, explicit invalid
controls and a software programming step driven by SPICE EqProp readouts.
Input `a` stays fixed during that learning check. The updated circuit is
simulated again in SPICE. Default execution needs only a CPU and takes seconds.

`tests/test_analog_energy_transformer.py` checks the electrical and mathematical
contracts. Set `ENERGY_TRANSFORMER_NGSPICE` to run its independent SPICE test.
Evidence characterization always requires SPICE; it never silently skips it.
