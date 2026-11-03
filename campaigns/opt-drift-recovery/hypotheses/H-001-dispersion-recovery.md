---
id: "H-001"
title: "Large drift dispersion creates a need for analog updates"
---
# H-001 — Large drift dispersion creates a need for analog updates

## Provenance and motivation

Prospective recovery hypothesis assigned by the user on 28 October 2026.
Prior drn evidence is linked in the [campaign overview](../README.md).
Historical fault and endpoint-error results motivate a test; they do not
validate this power-law drift claim.

## Claim and regime

With a frozen std3 OPT-125M source and scalar-coefficient power-law drift,
there is a high-dispersion regime where joint analog/digital recovery reaches
a predefined quality target within budget, while calibration, refresh and
digital-only recovery do not. Low-sigma and no-impairment controls distinguish
this from generic extra training.

Use the same aged chip, calibration starting point, data and digital parameter
scope across paired learning arms. Only joint recovery changes conductances.
The claim is limited to the chosen model, mapping, ages, exponents, methods and
budgets, separately for DRNs and crossbars.

## Support, falsification and inconclusive outcomes

Before held-out evaluation, freeze target loss L_target, minimum useful paired
improvement delta_min, cost limits, selected conditions, chip count and
uncertainty analysis in the experiment notes.

Support requires useful joint improvement and target attainment at high sigma,
failure of eligible alternatives to reach that target within budget, and
confirmation across independent held-out chips. Plot the paired effect versus
sigma; scope any boundary to the measured conditions.

The necessity claim is contradicted where calibration, refresh or digital-only
recovery reaches the target at acceptable cost. Better joint quality alone is
evidence of benefit, not necessity. Uniform benefit at low and high sigma does
not establish a special high-dispersion boundary.

Missing controls, inadequate precision, invalid solves, or all methods failing
to reach the target leave the intended recovery claim unresolved. Keep failures
and negative effects. A finite comparison establishes necessity only among the
tested alternatives; BPTT results do not validate a physical gradient circuit.
