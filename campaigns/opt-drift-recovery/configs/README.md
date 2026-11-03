# Native drn configuration handoff

The campaign uses the existing public execution surface and strict native
configuration format. This directory will hold exact configurations after
the shared scientific settings are resolved and the needed extensions exist.
It currently contains no executable lifecycle or placeholder scientific JSON.

| Component | Existing implementation |
| --- | --- |
| Coupled DRN and BPTT | [model/resistive/differentiable.py](../../../model/resistive/differentiable.py) |
| OPT wrapper | [model.py](../../../experiments/opt_cmo_drn/model.py) and [decoder.py](../../../experiments/opt_cmo_drn/decoder.py) |
| Physical state | [physical.py](../../../experiments/opt_cmo_drn/physical.py) |
| Recovery loop | [recovery.py](../../../experiments/opt_cmo_drn/recovery.py) |
| Native contract | [config.py](../../../experiments/opt_cmo_drn/config.py) and [existing configurations](../../../examples/opt_cmo_drn/) |

## Required work

Add the generic scalar-coefficient power law at the physical-conductance boundary and extend the native opt_cmo_drn configuration with explicit m, s, alpha, beta, time and state-replay fields. Preserve the same mathematical law and random-state contract as the crossbar backend, with the DRN's own physical device identities.

The current RecoveryStep freezes the model and enables conductance or calibration parameters; it also evaluates the teacher during recovery. Extend it for exact joint/digital-only suffix masks and stored clean post-HWA logits. The digital-only path must keep conductances unchanged while propagating input derivatives through the coupled DRN to trainable digital parameters. The joint path must train every eligible physical conductance and those same digital parameters.

Keep BPTT as the initial simulation reference. Calibration uses physically accessible ports; internal solver voltages are diagnostics. Retain the native public ebl train/validate entry points and artifact bundles. Record exact solver settings and any observed numerical failure; do not silently tune solver depth or amplification between arms.

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
