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
