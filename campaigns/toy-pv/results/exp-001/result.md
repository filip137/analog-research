# exp-001 result: contradicted

Experiment: [Does verify-and-retry reduce 1 h RMS error?](../../experiments/exp-001.md)

## Summary

The rule gives "contradicted": neither retry arm cut the 1 h RMS error by 5% or more against baseline. retry1 cut it by 2.9% and retry3 by 4.8%. Retrying does cut the 1 s error a lot (0.0294 to 0.0147 with retry3), but most of that gain is gone by 3600 s. That fits the note's reading that programming error is not what limits 1 h error and drift should be examined next. Seed SDs of the 3600 s means are small, between 7e-5 and 2.5e-4, compared with the differences between arms.

## Decision rule applied

I used the means of rms_err["3600"] over 3 seeds. The SDs are sample SDs; I recomputed the baseline SD from the per-seed values, and the retry1 and retry3 means match their per-seed values.
- baseline: 0.0794766 (sd 9.63e-5); extra_writes 0.0 (sd 0.0)
- retry1: 0.0771814 (sd 2.47e-4); extra_writes_per_device 0.4935 (sd 0.0064)
- retry3: 0.0756459 (sd 7.44e-5); extra_writes_per_device 0.8654 (sd 0.0017)

rel = 1 - arm/baseline:
- retry1: 1 - 0.0771814/0.0794766 = 0.00229522/0.0794766 = 0.0289 (2.89%)
- retry3: 1 - 0.0756459/0.0794766 = 0.00383072/0.0794766 = 0.0482 (4.82%)

Supported: needs some arm with rel >= 0.10 and extra_writes <= 0.5. retry1 passes the write limit (0.4935 <= 0.5) but rel is only 0.0289. retry3 fails both (rel 0.0482 < 0.10; writes 0.865 > 0.5). Supported does not apply.

Contradicted: needs every retry arm to have rel < 0.05. retry1: 0.0289 < 0.05, yes. retry3: 0.0482 < 0.05, yes. Contradicted applies.

Inconclusive is not reached.

## Concerns

1) retry3's rel of 0.0482 is close to the 0.05 cutoff. The 0.0018 gap is about 1.4e-4 in absolute RMS, roughly 2 standard errors of the difference given the seed SDs. If seeds are compared by index (the packet doesn't say they are paired), retry3's per-seed rel values are about 0.0475, 0.0501 and 0.0470, so seed 2 would just clear 0.05. The rule is stated on means, so the verdict holds, but it is borderline for retry3. 2) metrics.json has "toy": true, so these are toy-simulation results in normalized conductance units, not device data. 3) retry1's extra_writes of 0.4935 is just under the 0.5 limit, but that doesn't affect the verdict because its rel is far below 0.10. The config in metrics.json matches the note as launched (n_devices 20000, 3 seeds, the same three arms).

## Validity guards

- pass: exit code 0
- pass: metrics.json present
- pass: metrics finite
- pass: note unchanged since launch

## Provenance

- Launch commit: `db2d37882a875abfb3f9a430d65448985fbcf25a`; note sha256 `5e8490f47cf0e8fd…`
- Authorization: authorized_by: filip137 (bootstrap of the loop test, 2026-10-31)
- Run directory: `/home/filip/analog-research/loop-runs/toy-pv/exp-001`
- Reviewer: fresh `claude -p`, read-only tools, cost $0.089; reviewed 2026-10-31T21:22:05+00:00
