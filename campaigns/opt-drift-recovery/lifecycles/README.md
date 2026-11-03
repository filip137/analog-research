# Native crossbar configuration handoff

The campaign uses the existing public execution surface and strict native
configuration format. This directory will hold exact configurations after
the shared scientific settings are resolved and the needed extensions exist.
It currently contains no executable lifecycle or placeholder scientific JSON.

| Component | Existing implementation |
| --- | --- |
| OPT model | [workflow/networks/opt_mlp.py](../../../workflow/networks/opt_mlp.py) |
| Existing retention | [training/inference_retention.py](../../../training/inference_retention.py) and [workflow/retention.py](../../../workflow/retention.py) |
| Calibration | [training/drift_compensation.py](../../../training/drift_compensation.py) and [workflow/compensation.py](../../../workflow/compensation.py) |
| Recovery | [workflow/onchip.py](../../../workflow/onchip.py) |
| Native contract | [workflow/lifecycle.py](../../../workflow/lifecycle.py) and [lifecycle documentation](../../../docs/crossbar_lifecycle.md) |

## Required work

Add the generic scalar-coefficient power law as an explicitly named retention model and expose independent m, s, alpha, beta and time parameters through the strict lifecycle. The existing CMO drift_scale changes both mean and spread; the Wan noise-scale field represents total endpoint error. Neither is the requested law.

Connect the saved aged state to compensation and physical recovery without regenerating its noise. Extend the existing digital parameter catalog to the exact joint/digital-only suffix masks; a calibration flag or a digital LoRA arm is not this control. Check that every programmable physical device intended by the study is eligible in the joint arm, including references when the mapping uses them. Preserve the architecture-specific clean post-HWA teacher cache and report native versus matched block-output calibration.

Use the existing workflow planner, public ebl campaign runner and collector after the native schema supports the selected model and arms. Keep the source plan's analog-SGD requirement explicit; existing Adam write paths require a declared scientific reconciliation before reuse.

## Shared physical checks

Use the [selected model](../../../../on-chip-learning/campaigns/drift/power-law-model.md).
Validate scalar, conductance-independent m/s; independent alpha/beta; t=0
identity; s=0 mean-only behavior; reference-time exponent invariance;
non-reference time scaling; clipping; saved random-state replay; and frozen-age
post-write behavior. Keep the initial Gaussian model separate from optional
bin-dependent, delay, read-noise and heavy-tail variants.

Check exact conductance units and window before using a saved source. Historical
CMO state arithmetic cannot silently stand in for the supplied array window.
Keep the no-impairment control separate from ordinary programming error.

Before a run, pin code including relevant dirty/untracked files, resolved native
config, shared protocol/model snapshots, checkpoint, teacher/cache manifests,
state identities and run/output locations. Numerical sweeps must pass native
schema and proportional readiness checks. Read the worktree AGENTS.md and
execution policy; a historical run or target does not supply this campaign's
scientific budget or live monitoring owner.

## Implemented time-survey path

The scalar power-law inference path is now implemented and tested. The paired [time-survey specification](../../../../on-chip-learning/campaigns/drift/time-survey.md) fixes the current parameters and source identities. The `drift-times` canary/survey configs in this directory (crossbar filenames start `opt125m-cmo-`) are executable native inputs. Both live canaries passed; full surveys are running. Earlier gaps concerning global-joint/digital-only recovery remain future work.
