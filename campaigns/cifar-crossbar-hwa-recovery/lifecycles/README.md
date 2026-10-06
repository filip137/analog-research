# Lifecycle definitions

Each file defines one experimental loop under the
[lifecycle contract](../../../docs/crossbar_lifecycle.md): question, devices and
defects, then deployment, HWA, program-and-verify and on-chip training. The file
name equals its `lifecycle_id`. Experiment or pilot notes link a lifecycle and own
its decision rule, budget, handoff and interpretation; a definition alone
authorizes no execution.

| Lifecycle | Technology | Role |
| --- | --- | --- |
| [cifar10-pcm-conv4-smoke](cifar10-pcm-conv4-smoke.json) | Gaussian PCM, no relaxation | Labelled 128-image smoke of the workflow path |
| [cifar10-pcm-conv4-drift-smoke](cifar10-pcm-conv4-drift-smoke.json) | Gaussian PCM, 1 h drift + compensation | Labelled smoke of relaxation-aware P&V |
| [cifar10-om-conv4-smoke](cifar10-om-conv4-smoke.json) | AIHWKit 1.1.0 OM, closed-loop P&V | Labelled smoke of the OM path |

The smokes mirror the V2 sweep canary settings (seeds, 128 images, one HWA epoch)
but are new lifecycle runs, not historical replays. Their outcomes are recorded in
the [workflow pilot](../../pilots/20261005-crossbar-lifecycle-workflow.md) and are
not scientific evidence. Validate every definition with `python -m workflow check`.
