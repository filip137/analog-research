# Architecture handoff: applying the HWA structure to DRN

Prepared 4 November 2026 from HWA `main` at `5b85e85`. This describes the
completed HWA architecture and proposes how to carry its ownership model into
DRN. It is a conceptual handoff for a future implementation, not a file-by-file
move plan or a new scientific protocol.

The central change was to give network construction, physical state, learning
methods and execution separate owners. Experiments compose these capabilities;
reusable capabilities no longer live inside whichever experiment first needed
them. DRN can adopt the same structure while retaining its equilibrium-based
forward model and its existing scientific contracts.

## What changed in HWA

Previously, reusable network and device implementation was reached through
historical experiment packages. Workflow instructions and executable lifecycle
code shared a namespace. Some physical laws lived in methods or execution
stages, and hardware construction inferred network roles from parameter names.
These arrangements made a local change depend on unrelated experiment details.

The refactor established the following ownership:

| Owner | Responsibility | Allowed project dependencies |
| --- | --- | --- |
| `core/` | Neutral configuration primitives, identities, hashes, serialization and cache I/O | No higher owner |
| `hardware/` | Device populations, physical laws, faults, conductance state, physical random streams and cost counters | `core` |
| `networks/` | Architecture, teachers, data, logical parameters, physical mapping and forward adapters | `core`, `hardware` |
| `methods/` | HWA, programming controllers, calibration, on-chip updates, optimizers and maintenance | `core`, `hardware`, `networks` |
| `ebl/` | Public CLI, registry, composed configuration, runtime stages, native bundles, scheduling and collection | All lower owners |
| `experiments/` | Characterization and supported research-family compositions | Reusable owners and execution support where needed |
| `campaigns/` | Questions, declared comparisons, exact lifecycle inputs, handoffs and interpretations | Public execution interface and reusable analysis |
| `workflow/` | Instructions for defining, preparing, executing and reviewing work | Documentation only |

The import direction matters more than the directory names. Hardware must work
without importing a network, training routine or campaign. A network must be
constructible without importing the CLI or a particular experiment. `core` is
for genuinely neutral machinery; domain code stays with its owner.

This also changed some interfaces. Networks now supply explicit physical grouping
metadata instead of making hardware interpret OPT parameter names. Native and
physical OPT forwards share one traversal, with an adapter supplying the physical
behavior. Physical drift laws belong to hardware; stages decide when to invoke
them. Collection reads and verifies artifacts through reporting code without
importing numerical stage execution.

The split is practical rather than absolute dependency injection: some composed
method routines still receive the immutable lifecycle. Hardware constructors,
however, receive explicit physical settings and mapping metadata.

## State and configuration follow the same boundaries

Moving functions is insufficient if several layers still mutate the same state
or reinterpret its meaning. HWA assigns each evolving quantity an owner:

| State | Owner and contract |
| --- | --- |
| Logical weights and digital master | Network; off-chip HWA updates this master using sampled hardware views |
| Persistent conductances, faults and held apparent state | Hardware; writes mutate physical state and refresh the touched cells' apparent state |
| Physical randomness, drift history and cost counters | Hardware; serialized wherever required for supported replay |
| Optimizer moments, accumulation and update steps | Method/writer; kept distinct from device state |
| Paths, source identities, publication and run status | Execution stage and native bundle |

Every device-facing forward, gradient, validation and selection uses the held
apparent state. Temporary HWA perturbations do not become permanent array faults.
A controller sees its declared observations; hidden device parameters remain on
the analysis side. These are interface contracts, not conventions left to each
training loop.

Configuration mirrors ownership: device settings live with hardware, architecture
and data settings with networks, and algorithm settings with methods. The lifecycle
schema composes these sections and checks their compatibility. The experiment
chooses their values. A shared template does not decide a new experiment's physics,
topology, dataset, metric or controls.

## Execution and research records have different jobs

`python -m ebl` is the public execution surface. Runtime stages assemble the
declared network, device and method, then publish native evidence. Scheduling and
cross-worktree orchestration invoke that surface through subprocesses. CLI
discovery and parser construction stay lightweight and do not import the numerical
stack.

The canonical runner checks source, configuration, inputs and artifacts before
reusing completed work. A directory marked complete is not enough. Collection
checks the evidence and exposes coverage; the research note owns interpretation.
Completion, validity and scientific sufficiency remain separate judgments.

`workflow/` explains the process. Campaign and pilot notes hold the question,
decision rule, budget and current handoff; strict configuration holds executable
settings; native bundles hold execution identity and measurements. Small records
are versioned, while large raw evidence remains ignored. Shared crossbar/DRN study
records live in the sibling `on-chip-learning/` tree; each backend owns its code
and native runs.

## How this translates to DRN

DRN already received a limited part of this migration: shared drift, calibration,
recovery and artifact utilities use the new owners. Its broader network and solver
architecture is unchanged. The current starting points remain `model/resistive/`,
`training/core/`, the other `training/` mechanisms and experiment compositions.
See the [DRN shared-code record](../../drn/docs/shared-native-architecture.md).

A suitable target keeps the same top-level owners, with the following DRN-specific
interpretation. These are proposed responsibilities, not claims that the migration
has already happened:

| DRN responsibility | Proposed home and boundary |
| --- | --- |
| Resistive topology, energy/interactions, perfect-diode equations and equilibrium solving | `networks/`, with a cohesive resistive-network subpackage; expose solving as part of the network's forward contract |
| Coordinate-descent implementation and convergence diagnostics | Beside the resistive equations; optionally an internal solver subpackage if reuse warrants it |
| Physical conductance populations, write laws, permanent defects and drift | `hardware/`; independent of DRN layer names and training objectives |
| Mapping held conductances into the loaded DRN | A network-owned physical adapter, supplying explicit device identities and grouping metadata |
| Gradient estimators, free/nudged-phase coordination, HWA modifiers, Tiki-Taka and optimizer state | `methods/`; methods invoke the network solver and hardware interfaces |
| Per-step training engine, guards and reusable probes | `methods/`; preserve the existing separation from lab-only runners |
| Experiment-specific assembly, epoch/stage execution and artifact publication | `ebl/` runtime, with supported family entry points retained as necessary |
| Reference harnesses and exploratory diagnostics | `labs/`; production methods must remain independent of them |

The important DRN distinction is the loaded equilibrium solve. HWA's crossbar
matrix multiplication with digital nonlinearities does not replace that solve.
Keep topology, amplification, boundary conditions, nonlinearities, convergence
criteria and update ordering intact. Solver settings remain part of the declared
scientific configuration even if the solver's implementation moves.

Separate solving an equilibrium from choosing the training phases and turning
their results into an update. The network owns the equations and solve; the method
owns the learning procedure. Network-owned differentiation helpers can remain next
to the equations when their correctness depends on those equations.

Likewise, distinguish an adapter's architecture from its training policy. DRN's
existing passive and digital low-rank structures belong with the network, while
their optimization belongs with methods. HWA's restrictions on its own comparison
arms do not retire supported DRN structures or select new DRN experiments.

## Compatibility to preserve

The HWA migration preserved numerical behavior, normalized configuration digests,
native JSON/tensor formats, seeds and physical accounting. It retired old Python
import paths and the executable `workflow` namespace. Historical runs remain
reproducible from their recorded source revision; frozen artifacts are not rewritten
to accommodate new code.

Use the same distinction in DRN: retain supported artifact and replay contracts,
update maintained callers, and keep compatibility entry points only for actual
consumers. CMO endpoints, OM pulses and PCM programming remain distinct models.
Architectural cleanup does not authorize changing their equations or filling gaps
in drift semantics.

HWA and DRN remain independently importable. The
[shared-module manifest](shared_native_modules.json) identifies the kernels that
must be byte-identical across the two trees. Coordinate changes to those copies
and verify the manifest; never solve sharing through sibling-worktree imports.
Shared source does not imply identical experiment settings.

Keep resume claims precise. HWA supports completed-stage reuse, completed drift
shards and certain historical epoch restores; it does not support general lifecycle
mid-stage resume. DRN must preserve and verify its own supported continuation
contracts, including solver or warm-start state where continuation depends on it.

## Suggested implementation sequence and acceptance

1. **Freeze the DRN baseline.** Identify maintained families and consumers, record
   source/config/input identities, and capture small deterministic forward,
   gradient, update and checkpoint traces. Inventory mutable solver, device,
   optimizer and random state before moving code.
2. **Assign owners before renaming packages.** Split mixed modules by responsibility.
   Extract neutral utilities and physical laws, then network construction/solving
   and reusable methods. Keep numerical changes separate and preserve operation
   and random-draw ordering.
3. **Make the boundaries explicit.** Pass mapping metadata to hardware and expose a
   DRN physical-forward interface to recovery methods. Reuse the established solve
   path across training, validation and selection. Keep realistic controllers
   limited to their declared observations.
4. **Consolidate execution.** Route maintained callers through the public CLI and
   canonical runner. Separate artifact collection from numerical execution and
   workflow documentation from runtime code. Preserve strict configuration and
   artifact identities during the transition.
5. **Validate the resulting system.** Enforce import directions, compare bounded
   traces and future state after restore, check shared modules, and exercise an
   isolated package installation. Update architecture guidance and active callers
   together; retain historical evidence at its pinned revision.

Acceptance should require exact replay for a pure relocation: outputs, gradients,
equilibria, convergence decisions, physical transitions, random state and cost.
Exercise representative digital and physical DRN paths, supported learning methods
and checkpoint continuation. A discrepancy needs an explanation; widening a
tolerance is not evidence that ownership changes preserved behavior.

Measure import cost and a small representative workload after correctness is
established. The HWA migration reduced CPU process memory through narrower imports;
it did not establish a kernel speedup or GPU improvement. Its validation does not
validate the proposed DRN migration, which needs its own bounded evidence.

Detailed HWA references: [architecture and ownership](architecture.md),
[migration evidence](../campaigns/pilots/20261103-architecture-boundaries.md),
[boundary checks](../tests/test_architecture_boundaries.py),
[execution policy](../workflow/policy.md) and
[lifecycle contract](crossbar_lifecycle.md).
