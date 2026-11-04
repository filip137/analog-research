# DRN ownership and execution

DRN separates physical state, network equations, learning procedures and execution.
The numerical model remains a loaded perfect-diode resistive network, including
its existing solvers, amplification, boundary conditions and low-rank structures.

| Owner | Responsibility | Project dependencies |
| --- | --- | --- |
| `core/` | Neutral configuration, identities, serialization and cache primitives | None |
| `hardware/` | Device populations, physical laws, conductance and apparent state, physical randomness | `core` |
| `networks/` | Architecture, teachers, data, mapping, equilibrium equations and solvers | `core`, `hardware` |
| `methods/` | Learning phases, gradients, optimizers, programming controllers, calibration and maintenance | `core`, `hardware`, `networks` |
| `ebl/` | Public CLI, composed schemas, runtime assembly, native bundles and scheduling | Lower owners |
| `experiments/` | Research analyses, characterization reports and supported launch compositions | Reusable owners and execution |
| `campaigns/` | Questions, declared comparisons, handoffs and interpretation | Public execution interface |
| `workflow/` | Execution instructions | Documentation only |
| `labs/` | Reference harnesses and exploratory diagnostics | Reusable owners; never imported by production mechanisms |

## Construction and state

`networks/resistive/` owns topology, energy interactions, perfect-diode equations,
coordinate-descent solving and passive/digital adapters. Small-network and MNIST
builders construct these objects independently of execution. `networks/opt/`
owns the coupled OPT architecture and physical adapter. Logical and physical OPT
forwards share `OptDrn.forward`; adapters supply held conductances and corrections.
Networks provide explicit physical keys, shapes, roles and fault-group metadata.
Hardware does not infer topology from OPT parameter names.

| State | Owner and rule |
| --- | --- |
| Logical parameters and digital masters | Network; methods update them through the declared learning rule |
| Equilibria, warm-start layers and convergence settings | Network solver; restored where supported continuation requires them |
| Persistent conductances, held apparent values, physical RNG and device assignments | Hardware; writes update persistent state and refresh apparent values for touched cells |
| Optimizer moments, EP phases, accumulation and learning counters | Method; distinct from the physical plant |
| Paths, input hashes, runtime identity and bundle status | Execution |

Every device-facing forward, gradient, metric and selection uses the held apparent
state. Off-chip HWA uses a sampled hardware view while updating the digital master.
Controllers receive their declared observations; hidden physical parameters stay
on the analysis side. Hardware trace banks own measured-curve assignments and pulse
indices; methods own projection or update policy. Physical checkpoint keys remain
unchanged across the ownership move.

`methods/core/` owns the production step engine, gradient estimators, optimizer,
modifier seam and guards. The lab epoch harness lives in `labs/training/`.
Domain settings live in `hardware/`, `networks/config/` and `methods/config/`;
`ebl/schema/` composes and validates them without importing numerical execution.

## Execution and compatibility

Use `python -m ebl` for training, validation, characterization and inspection.
`ebl/runtime/` assembles family components and publishes native artifacts.
`ebl/execution/runner.py` owns subprocess campaign execution and verified reuse.
`ebl/lifecycle/` owns preparation, collection, staging and scheduling. For example:

```bash
python -m ebl lifecycle plan --help
python -m ebl lifecycle collect RUN_DIR --output collection.json
python -m ebl lifecycle recovery --help
python -m ebl campaign --help
python -m ebl runs inspect RUN_DIR --verify-artifacts --require-complete
```

CLI discovery and collection do not import Torch, NumPy, HDF5 or AIHWKit. External
OM sampling and its source/artifact receipts belong to `ebl/runtime/device_sampling.py`;
the native device sampler is `hardware/ibm_om/sampling.py`. Execution injects that
service into methods that need a persisted population artifact.

Retired Python namespaces are `model`, `training` and executable `workflow`.
Import each defining owner directly; there is no blanket compatibility facade.
Existing strict configs, normalized digests, native tensor/JSON schemas, seed order
and supported checkpoint continuation remain contracts. Historical runs use their
pinned source revision. No general mid-stage resume capability is added.

The [shared-module manifest](shared_native_modules.json) names 15 byte-identical
HWA/DRN files. Each checkout imports independently. This migration does not add
temporal drift correlation or CMO endpoint draws on rewrite; those remain separate
physics work governed by AGENTS.md.

## Validation and extension

`tests/test_architecture_boundaries.py` enforces import directions, lightweight
discovery, independent construction, configuration identity, shared bytes and exact
OPT forward/gradient traces. Existing OM, characterization, training-loop and full
lifecycle goldens check numerical and continuation contracts without recapture.
The [migration pilot](../campaigns/pilots/20261104-architecture-boundaries.md)
records commands, coverage and limitations.

Add equations and builders under networks, device behavior under hardware, and
learning policy under methods. Compose them in a versioned `ebl/schema/<family>.py`
and `ebl/runtime/<family>/`, then register the capability in `ebl/registry.py`.
Keep scientific choices in exact configs and campaign/pilot records. Follow the
[execution policy](../workflow/policy.md) for authorization and lifecycle evidence.
