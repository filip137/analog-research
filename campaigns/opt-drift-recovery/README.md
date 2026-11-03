# OPT drift dispersion and drn recovery

**Direction update, 30 October 2026:** the shared
[drift mitigation campaign](../../../on-chip-learning/campaigns/drift-mitigation/README.md)
now owns new work. Qualify the device model and establish strong HWA/P&V
baselines before further on-chip learning comparisons. The completed
[custom recovery pilot](../../../on-chip-learning/campaigns/drift/recovery-pilot.md)
and earlier results remain preliminary evidence with their original contracts.
The defaults and unfinished experiments below are the historical plan, not a
new execution assignment.

Does a large standard-deviation term in drift make on-chip analog updates
necessary to recover OPT-125M quality beyond calibration, refresh and
digital-only recovery within the declared budgets?

**State: preliminary evidence retained; follow-up learning depends on the new campaign.** This is the drn half of the shared
[dispersion study](../../../on-chip-learning/campaigns/drift/dispersion-study.md),
paired with the [other architecture](../../../hwa/campaigns/opt-drift-recovery/README.md).
The user selected a generic power law with scalar mean/spread coefficients,
one-hour reference time, and configurable exponents (0.23 and 0.11 nominal).
The [model specification](../../../on-chip-learning/campaigns/drift/power-law-model.md)
owns its equation and refinements. Conductance dependence is deferred.

Use frozen std3 HWA, six analog MLPs in decoder blocks 7–12, the fixed
1,024-sequence recovery subset, and the shared 128-update pilot allowance.
Recovery compares only `global_joint` and `digital_only`, with the same
digital scope, cached end-to-end targets and data allowance. Local recovery
arms remain deferred. These are explicitly assigned generic simulations.

- [Hypothesis H-001](hypotheses/H-001-dispersion-recovery.md)
- [Series 001](series/001-dispersion/README.md)
- [First experiment](series/001-dispersion/experiments/exp-001-dispersion-calibration.md)
- [Native configuration handoff](configs/README.md)
- [Generated ledger](ledger.md)

Prior evidence: [OPT DRN HWA/recovery pilot](../pilots/20261016-opt125m-cmo-drn-hwa-first.md). Existing OPT DRN work supplies the coupled forward, BPTT and physical recovery components. Its fault-recovery results do not establish recovery under this power-law drift model or the new joint/digital-only comparison.
The recovery question remains open; the [time-survey result](series/001-dispersion/results/exp-001.md) records the completed degradation grid.

The completed subphase is the [time-to-degradation survey](../../../on-chip-learning/campaigns/drift/time-survey.md): find manageable degradation at 24 hours and large degradation at one month or one year. It evaluates frozen states before calibration or recovery. Scalar drift support, local checks and both live cluster canaries passed. Both full surveys were validated and reviewed: no tested setting satisfies both thresholds. Calibration and recovery remain deferred.
