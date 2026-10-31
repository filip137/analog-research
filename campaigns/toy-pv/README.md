# toy-pv: experiment-loop plumbing test

**This campaign exists to test the [experiment loop](../../loop/README.md). Its
simulator is a toy; nothing here is device evidence.**

## Question

With a fixed toy device (write noise, conductance-dependent log-time drift,
conductance clamped to [0, 1]), which programming scheme gives the lowest RMS
conductance error one hour after programming, and at what extra-write cost?

## Knobs

[envelope.json](envelope.json) lists what the loop may vary and the allowed ranges:

- **Verify-and-retry:** `max_retries` rewrites of devices whose first read is
  outside `±tol` of the aim.
- **Drift pre-compensation:** with `precomp_t_ref_s` set, aim at the conductance
  expected to drift onto the target at that age; `null` aims at the target.

The device model in [runner.py](runner.py) is fixed and is not an experimental
variable.

## Metrics

- Primary: `rms_err["3600"].mean`, the RMS error from target at 3600 s,
  in normalized conductance, averaged over seeds.
- Cost: `extra_writes_per_device.mean`.
- Secondary: `rms_err["1"].mean`, the quality at first read.

Arms within one experiment share targets per seed, so their comparisons are paired.

## Direction

Find the scheme with the lowest 1 h error at no more than 0.5 extra writes per
device, without making first-read error worse than the baseline. Stop when
further experiments within the envelope would not change that choice.
