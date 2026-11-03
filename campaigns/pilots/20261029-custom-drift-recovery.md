# Custom-drift recovery and repeated maintenance

Status: implementation and bounded Jean Zay timing canary; full pilot unreleased.

The shared scientific specification, budget, evidence and monitoring handoff are
in [recovery-pilot](../../../on-chip-learning/campaigns/drift/recovery-pilot.md).
This worktree uses its native drift_recovery stage and pinned std3 HWA source, six
analog OPT-125M MLPs, frozen digital parameters and custom additive device drift.
Only the two 30-minute development canaries are authorized for this release.

The canary and full pilot configurations are in this tree's
`campaigns/opt-drift-recovery/` directory. Full execution requires the allocation
decision after measured timing/storage estimates. Canary accuracy is smoke evidence.
