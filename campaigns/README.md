# Research questions and campaigns

- [Crossbar HWA and array-specific recovery on CIFAR](cifar-crossbar-hwa-recovery/README.md):
  dense CIFAR-10 and ResNet-32 CIFAR-10/100, OM and PCM, with imported results,
  source provenance, reproducible reports and a separate migration audit.

Start small: one note in `pilots/<question>.md` owns a bounded question through review.
Crossbar loops are specified as strict lifecycle definitions in `<campaign>/lifecycles/`
([contract](../docs/crossbar_lifecycle.md)); the note links its lifecycle and keeps the
decision rule, budget, handoff and interpretation.
Use the [pilot template](../.agents/skills/experiment-loop/references/pilot.md).
Promote sustained work into `<campaign>/README.md`, optional explorations/hypotheses,
and series containing experiments/results. Preserve original pilot evidence by link.
See the [workflow](../docs/experiment_workflow.md) and
[record conventions](../.agents/skills/experiment-loop/references/artifacts.md).

The existing `runner.py`, `schema.py` and `manifests/` are executable subprocess
orchestration, not a mandatory research-planning hierarchy. Historical JSON plans
remain under `studies/`; large artifacts remain under `results/`.

Historical workflow validation: [local pilot](pilots/20260925-hwa-bounded-learning.md).
Current execution scope and compatibility checks:
[crossbar cleanup](pilots/20261004-crossbar-worktree-cleanup.md).
