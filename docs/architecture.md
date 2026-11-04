# Native architecture and ownership

The migration separates network construction, physical state, algorithms and
execution. `workflow/` is [agent documentation](../workflow/README.md).
`ebl` is the public execution interface. Scientific settings and native bundle
identifiers remain stable; this is not a new device regime.

The [DRN architecture handoff](drn-architecture-handoff.md) explains the design
conceptually and proposes how to apply the same ownership boundaries to DRN.

## Ownership and allowed dependencies

| Owner | Responsibility | May depend on |
| --- | --- | --- |
| `core/` | Neutral configuration primitives, parameter catalogs, hashes, artifact bytes, tensor serialization/cache I/O | Standard and numerical libraries |
| `hardware/` | Matrix identities, device populations and laws, permanent faults, persistent/held state, physical RNG and counters | `core` |
| `networks/` | OPT/CIFAR construction, teachers, data/cohorts, analog subsets, logical mapping and forwards | `core`, `hardware` |
| `methods/` | HWA, on-chip adaptation, calibration, programming/verify, maintenance, digital optimizer and checkpoint state | `core`, `hardware`, `networks` |
| `ebl/` | Discovery, strict lifecycle composition, stage assembly, bundle store, processes, scheduling, recovery and collection | All lower owners; reference entry points through the registry |
| `experiments/` | Native device characterization and supported historical reproduction entry points | Reusable owners; execution support where required |
| `campaigns/` | Scientific notes, immutable lifecycle inputs and thin assigned launch/analysis scripts | Public CLI and reusable analysis |

`tests/test_architecture_boundaries.py` enforces the lower four import directions.
Domain configuration lives in each owner (`hardware/config.py`, `networks/config.py`,
`methods/config.py`); `ebl/schema/lifecycle.py` composes and cross-validates it.
The immutable lifecycle is still passed to several composed method routines.
Hardware constructors receive explicit physical settings and fault-domain metadata,
not a lifecycle or an OPT naming convention. CLI discovery/parser construction stays
stdlib-only. `ebl/runtime/store.py` owns run-store study lookup; `core/artifacts.py`
does not own orchestration.

## Active execution paths

| Path | Concrete owners and entry points |
| --- | --- |
| OPT construction/loading | `networks/opt.py`: `OptConfig`, `OptDecoder`, `from_hugging_face`, `load_teacher`; `networks/deployment.py`: `load_teacher`, `build_network` |
| Analog selection and logical identities | `OptMlpSuffix.__init__` selects suffix MLP `fc1`/`fc2`, constructs `MatrixSpec` records and digital remainder; `networks/mapping.py: fault_domains` declares physical grouping/roles |
| Native and physical OPT forwards | `OptMlpSuffix.forward_features` and `forward_suffix`; `networks/opt_drift.py: CrossbarAdapter.forward` performs physical encoding then delegates the same suffix traversal |
| HWA and digital master | `methods/hwa.py: train_source`, `endpoint_sampler`, `selection_score`, `train_lora`; temporary views are drawn from explicit streams while the FP32 master is optimized digitally |
| Deployment/programming/faults | `methods/programming/lifecycle.py: fresh`, `configure`, `program`; `hardware/arrays.py: fresh_array`, `hierarchical_faults`, `array_identity`, `verify_fixed`; `hardware/devices/{cmo,pcm,om}.py` own the distinct device laws |
| Held state and drift | `hardware/devices/cmo.py: CmoMatrix`; `hardware/drift/{writable,power_law,inference,pcm,retention,flowers,affine}.py`; fitted digital readout operators in `hardware/readout.py`, fitting algorithms in `methods/calibration/` |
| On-chip updates | `methods/onchip.py: train_arm`, `make_writer`, `IdealizedWriteAdam`, `ThresholdedWriteAdam`, pulse writers; shared maintenance in `methods/maintenance/recovery.py` and optimizers in `methods/optimizers/` |
| Lifecycle assembly | `ebl/runtime/lifecycle.py: Stage`, `prepare`, `devices`, `train_hwa`, `deploy`, `train_onchip`, `run_train`; dedicated retention/compensation/recovery stages compose lower owners |
| Native characterization | `experiments/reram_program_verify/`; `ebl/runtime/device_sampling.py` and `sampling.py` own subprocess isolation and source receipts |
| Checkpoint and reuse | Physical classes serialize their state/RNG/counters; `methods/checkpoint.py` owns historical training codecs; `ebl/execution/runner.py: run_campaign` checks stage fingerprints and artifacts; `recovery_shards.py` continues completed shards |
| Collection | `ebl/lifecycle/collect.py`, pure `*_rows.py` serializers; `ebl/execution/lifecycle_collect.py` and `recovery_collect.py` verify slices without importing numerical stage execution |

For OPT-125M, each analog decoder MLP has 4,718,592 logical weights. Attention,
nonlinearities, biases, LayerNorm, embeddings and the tied LM head remain digital
under the supported suffix contract. A different analog subset remains a scientific
choice, even though its implementation now has a clearer home.

## Mutable state and replay

The network owns trainable digital parameters and the logical master. A device or
array owns persistent conductances, permanent fault identities, held apparent reads,
physical random streams and physical counts. Writers own digital moments and update
step counts; drift objects own ages, episode noise and their random state. The stage
owns paths, source provenance, artifact publication and wall-clock observations.

HWA minibatch noise and temporary corruption never become a deployed fault map.
Device-facing gradients and selection use held post-write apparent state. The
existing code's drift timing, write/reset rules and coupling are preserved, including
the fixed-episode historical drift implementations and exact-endpoint writable law.
This migration does not implement a new temporally correlated or CMO-write drift
model. A chosen experiment must check that its declared physics is implemented.

CMO endpoint writes, OM pulses and PCM programming/retention remain separate laws.
Their physical costs are not interchangeable. Tensor hashes with shape/dtype metadata
and historical raw-byte hashes remain distinct format contracts. Restricted checkpoint
loading no longer retries an unrestricted load after `TypeError`.

Native JSON/tensor state keys and schema strings remain unchanged. Old Python import
paths and the workflow CLI are retired. Frozen source archives and scientific records
are unchanged: replay them from their recorded revision. Full lifecycle mid-stage
resume is still unsupported; epoch checkpoints in supported historical families,
completed stage reuse and completed drift-shard continuation are distinct capabilities.

## Conflicts and resolutions

| Conflict and observable risk | Root cause / resolution | Affected callers and check |
| --- | --- | --- |
| `workflow` meant both agent instructions and executable science | Mixed ownership; move code into its owners and CLI into `ebl lifecycle` | Maintained CLI/scripts, skill and docs; CLI, package and lifecycle tests |
| Shared plants/network code was reached through reference experiments | Reusable implementation lived under a historical family; extract it without changing equations | Active and supported historical callers; reference equivalence and full suite |
| Hardware inferred MLP and adapter roles from OPT-like strings | Network semantics in fault construction; pass explicit `FaultDomain` metadata | Every `hierarchical_faults` and `fresh_array` caller; 12 exact fault-map/receipt comparisons and arbitrary-name regression |
| Native and physical OPT suffix traversals were duplicated | Two implementations could drift; one `forward_suffix` traversal, explicit physical adapter | Native/maintenance forwards; 10 exact logits/gradient/capture comparisons |
| PCM relaxation and retention equations lived in methods/stages | Physical law mixed with invocation; move laws/state into `hardware/drift`, retain stage composition | Programming, retention, compensation; numerical/replay tests |
| Collector imported private CLI/numerical stage code | Reporting coupled to execution; pure row serializers and source-fit validation | Native and subprocess collection; collector and stdlib parser tests |
| Campaign stage helper skipped any complete run | A separate runner omitted identity checks; remove it, support selected stages and explicit bindings in the canonical runner | Local/Slurm launch kit, artifact lookup; changed config/input/source tests and read-only artifact test |
| Checkpoint `TypeError` retried unrestricted loading | Broad compatibility fallback hid loader errors; require the restricted loader | Historical codecs; focused failure test proves no second load |
| Shared HWA/DRN drift files could diverge during extraction | Independent distributed sources; explicit mirror manifest and byte comparison | DRN callers/packaging, 15 shared modules; `benchmarks/verify_shared.py` and DRN replay tests |

These are architectural conflicts. Differences between CMO, OM, PCM, named drift
laws, or held-state and diagnostic persistent-state views are intentional contracts.
No physical laws are unified by this change.

## Retained boundaries and adapters

- `CrossbarAdapter` is permanent: shared physical recovery needs a network-specific
  bridge for physical layout, forward and capture; it delegates the common traversal.
- The native AIHWKit subprocess boundary is permanent: separate environments retain
  explicit population provenance and the serializable native pulse plant.
- Independent HWA/DRN copies are a supported distribution contract, recorded in
  [shared_native_modules.json](shared_native_modules.json). Verify them with
  `python -m benchmarks.verify_shared <hwa-checkout> <drn-checkout>`; no cross-worktree imports.
- Supported historical codecs and family entry points remain for real reproduction
  consumers. They are not blanket aliases for removed packages.
- [The old policy URL](experiment_workflow.md) is a permanent documentation redirect
  because frozen records link to it. There are no transitional Python import facades.

AIHWKit 1.1.0's separate `nn.modules`, `simulator.parameters.mapping`, simulator
devices/tiles and `optim` packages informed the ownership split. This is a design
analogy, not adoption of its native tile state or replay semantics. The reference is
the locally installed versioned source, also published in the
[AIHWKit source tree](https://github.com/IBM/aihwkit/tree/v1.1.0/src/aihwkit).

Measurements, exact comparisons, test results and remaining limitations are in the
[migration pilot](../campaigns/pilots/20261103-architecture-boundaries.md).
