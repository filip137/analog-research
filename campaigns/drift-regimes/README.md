# DRN drift-regimes: reduced matched screen

This branch prepares **exp-003**, the reduced DRN counterpart of the crossbar
experiments completed on 4 November 2026. The user requested a new worktree,
branch and experiment plan on 5 November. This is the planning checkpoint;
no GPU job, timing probe or numerical implementation has been started.

The shared scientific record is authoritative:

- [Experiment plan](../../../../on-chip-learning/campaigns/drift-regimes/series/002-regime-map/experiments/exp-003-drn-matched-screen.md)
- [75 exact crossbar counterparts](../../../../on-chip-learning/campaigns/drift-regimes/series/002-regime-map/comparisons/exp-003-crossbar-counterparts.csv)
- [Read-only counterpart audit](../../../../on-chip-learning/campaigns/drift-regimes/series/002-regime-map/code/exp-003-counterparts.py)

These files were committed in the separate `on-chip-learning` repository on
branch `plan/drn-drift-regimes-slim`, at
`15512d50d697d8174604a689b0c6ef0484ddede8`. This pointer does not duplicate the
scientific plan. Update that record when the design changes.

## Checkout and evidence references

| Item | Identity |
| --- | --- |
| This branch | `drn/drift-regimes-slim` |
| Worktree | `/home/filip/analog-research/worktrees/drn-drift-regimes-slim` |
| DRN base | `refactor/training-core`, `2f0e0750deda4d31d40b8198c6bbf8f389550428` |
| Crossbar source reference | `exp/drift-regimes`, `3edcaf63a0d41d489b12c7bccf93d2dfb5c4653b` |
| Earlier DRN family mirror | `drn/drift-regimes`, `ce61897` — preserve; it predates the current code layout |
| Future native results | `/home/filip/analog-research/drn/results/drift-regimes/exp-003/` |

The plan selects 25 method–regime cells, each with three seeds: 69 primary
and six diagnostic trajectories, plus five separate R0 LayerNorm tuning
trajectories. LoRA is deferred at the user's request.
Every comparison has a collected crossbar counterpart. The shared audit
checks complete age grids, file hashes, common inputs, paired release states,
executed arm settings and the 24-hour KL integral.

## Implementation handoff

Read the shared plan before implementing. Its **5–150 uS drift bounds**,
**9–88.2 uS programming bounds**, current crossbar Fisher-rewrite semantics,
common FP32 teacher and WikiText-2 cohorts explicitly distinguish this study
from the older defaults and recovery studies described elsewhere in this tree.

Use the existing ownership and public `python -m ebl` surface:

- `hardware/drift/`: matched drift-family state and counter-keyed replay.
- `methods/`: P&V, calibration, adaptation and physical rewrite controllers.
- `networks/opt/` and `networks/resistive/`: coupled apparent-state forwards,
  LayerNorm gradient propagation and physical-coordinate Fisher sensitivities.
- `ebl/schema/opt_cmo_drn.py` and `ebl/runtime/opt_cmo_drn/`: strict native
  configuration, event timing, accounting and native bundles.

The existing DRN recovery schema is not a runnable configuration for this
screen. The shared plan records the proposed eight-sweep train/evaluation
policy and the four-device Fisher extension. LoRA, its tuning and the
adapter-interface work are outside this screen; no LoRA placement decision
is needed to proceed with the retained methods.

Execution remains unassigned. Before a future launch, finish the native
implementation and proportional readiness checks, pin configs/input paths
and a launch commit, measure runtime, and record a
target, GPU-hour/storage caps and monitoring owner in the shared record.
Keep raw artifacts under the declared results root outside this worktree.
