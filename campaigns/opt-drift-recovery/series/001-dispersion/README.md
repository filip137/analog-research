# Series 001: dispersion, recovery and time-exponent sensitivity

Use the [shared power law](../../../../../on-chip-learning/campaigns/drift/power-law-model.md)
with constant m and s across devices. Begin by varying s at fixed mean,
age and nominal exponents. Then test the effect of alpha and beta at
non-reference ages; at one hour both uncorrected power factors are one.

| Experiment | Purpose | Current state |
| --- | --- | --- |
| [exp-001](experiments/exp-001-dispersion-calibration.md) | Find residual loss after calibration as sigma increases | Planned |
| [exp-002](experiments/exp-002-joint-digital-recovery.md) | Compare joint, digital-only and refresh at selected conditions | Planned |
| [exp-003](experiments/exp-003-exponent-sensitivity.md) | Test mean/spread exponent sensitivity on development conditions | Planned |
| [exp-004](experiments/exp-004-heldout-confirmation.md) | Confirm the frozen comparison on independent held-out chips | Planned |

Keep sigma, age, exponent setting and chip identity separate in result tables;
do not pool them into one average recovery score. Report clipping and realized
mean shifts alongside dispersion. Different programming seeds on one assignment
are not independent chips. Raw bundles belong under
`results/opt-drift-recovery/exp-00N/`; reviewed result notes will be added to
this series' `results/` directory when evidence exists.
