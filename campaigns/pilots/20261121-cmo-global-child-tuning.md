# Global Child-Tuning-D under CMO drift on XB6 and XB12

Study record: [on-chip-learning #67](https://github.com/filip137/on-chip-learning/issues/67).
Compare the fixed global 1% and 10% Fisher masks with the original per-matrix 10%
arm on each two-sided crossbar at CMO ×1, ×4 and ×10. The decision compares
24-hour trapezoid mean teacher KL, endpoint KL and device writes. The original
configuration supplies the source, device state, sampled targets, evaluation
cohort and learning rate. The global population is the 28,311,552 or
56,623,104 logical weights over all analog matrices. Both physical legs of
each selected logical weight follow the original writer.

Each of the six full lifecycles runs untreated, the release twin and the two
2,400-update learners. At ×4 each model also runs both drift-free learners.
The two 32-update ×4 canaries cover both fractions, drift-off controls and
write-audit publication. Original CAL is the zero-write trained control and
can be reused after exact release and untreated matching. Neither branch
runs the separately scoped learning-rate brackets.

The pre-submit limit is 4.3 GPU-h of the cumulative 36 GPU-h issued to the
shared study. Both canaries reserve 15 minutes each. Full allocations reserve
25/40 minutes for XB6 outer/×4 and 35/55 minutes for XB12 outer/×4.
Use the original Jean Zay V100-SXM2-32GB execution resources, account
`umg@v100`, partition `gpu_p13`, QoS `qos_gpu-dev`, ten CPUs and PyTorch
2.5.0. The previously measured runtimes support these limits. The original
XB6 learner ran in about 570 seconds and XB12 in about 790. The native
`lifecycle describe` report shows four stages before frozen-source reuse
and twelve or twenty-four analog matrices. There is no new HWA training.

## Readiness and handoff

- Lifecycles in `20261118-cmo-drift-recovery/global-child/{canary,full}`.
- Native flow uses `campaigns/fault-recovery-paper/launch/` source staging,
  `jz_prepare.sh`, `jz_submit.sh`, `run.slurm` and `worker.py`.
- The worker verifies the pooled selection count, per-matrix partition, shared
  same-fraction mask hashes within a lifecycle, and zero writes outside the
  mask. The launch manifest binds the original deployed-state identities and
  exact untreated and release-twin KL grids per unit. The live canaries must
  reproduce both controls before full submission.
- CPU gate: 66 focused tests passed. The workflow suite in a separate process
  passed 694 tests with five skips. Native lifecycle check and describe passed
  for all eight configs. Shell syntax and Python compilation passed.

The initial V100 canaries, Slurm 554400 and 554401, completed native runs and
mask audits. The final historical control assertion rejected the new
`clipped_targets_low` and `clipped_targets_high` report fields. The exact
deployment, data and grids all matched. The historical clipped-target totals
were 9 (XB6) and 21 (XB12), entirely on the low side. The comparison now checks
that the new counts are nonnegative integers summing to the historical total
before comparing the remaining report exactly. The failed jobs charged 345
and 418 seconds respectively. Rerun both live gates on the new worker at
12 minutes each. Keep 12,900 seconds reserved for the six full jobs; the
first charges plus both ceilings total 15,103 seconds of the 15,480-second
HWA budget.

## Live V100 handoff

The pinned backend checkpoint is `9319f9d554631e734afa0279d6b85c4cf20bb789`.
Jean Zay staged and verified 2,583 source files. The launch directory is
`$WORK/cmo-drift-recovery-global-child-20261121/launch-9319f9d`. Its
`manifest-xb6.json` and `manifest-xb12.json` pin the source receipt, native
configs and exact original controls. Output roots are
`$SCRATCH/cmo-drift-recovery-global-child-20261121/runs-retry-v1/xb{6,12}`.
The exact
staged plans and Slurm test-only checks passed for all eight configs.

The replacement canaries 554711 and 554712 passed on the original V100
contract. Their 32 updates per arm produced nonzero writes within each mask
and zero outside. XB6 selected 283,116 and 2,831,155 weights. XB12 selected
566,231 and 5,662,310 weights. Both masked learners on a given fraction used
the same mask as their drift-free controls. Untreated and release-twin
trajectories match the original at all ten ages. The retries charged 337
and 424 seconds. The combined canary charges are 1,524 seconds.

| Platform | ×1 | ×4 | ×10 |
| --- | ---: | ---: | ---: |
| XB6 | 554883 | 554884 | 554885 |
| XB12 | 554887 | 554888 | 554889 |

The six full jobs reserve 12,900 seconds. Full reservations plus the charged
gates total 4.006666667 GPU-h of the 4.3 GPU-h backend allocation. The shared
study owns the cumulative ledger. The no-drift controls are trained once per
model and fraction in the ×4 job. The native worker writes `complete.json`
after artifact and control validation. The shared root retains the durable
watch, job table, collected certificates and callback under
`artifacts/cmo67-global-child-xb/`. Validate each completed native run again
with `ebl runs inspect --verify-artifacts --require-complete`, confirm
Slurm accounting and compare mask hashes across multipliers before analysis.
