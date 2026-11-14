# exp-002 (crossbar): learning-rate screen with all 12 MLPs analog, 3 % random stuck devices, tuning chip 271001

State: implemented; deployment being pinned (handoff below). Branch: `exp/fault-paper-tuning-l12`.

**Authorization (user, 14 November 2026):** the 12-MLP version of the exp-001 screen, run
locally on the RTX 3090 `nom-cool-1` (idle at start; one GPU, arms sequential).

## Question and decision

Which learning rate does each crossbar recovery arm select at 3 % random stuck devices when
fc1/fc2 of all 12 decoder layers are analog (tuning assignment 271001)? Same rule as
[exp-001](exp-001-crossbar-lr-screen.md): final endpoint, lowest development teacher KL; ties
within 1 % → fewer writes; a selection at a grid edge is reported, not extended.

## Frozen comparison

Everything is exp-001's frozen comparison (its table and arm definitions) except:

| Setting | exp-002 |
| --- | --- |
| Network | `network.analog_decoder_layers = 12`: fc1/fc2 of decoder layers 0–11 (56,623,104 weights, 24 matrices) |
| HWA source | the 12-MLP std3-recipe fit `ed53b6cbe2a7315f364291c7142ee2826c6ca4a44c21a2d5d25190b8c90569b0` of drift-all-mlps (same `hwa` section as exp-001; declared as in `campaigns/drift-all-mlps` lifecycles) |
| Teacher samples | every unit (deployment, B scale, canary, screen) loads the stored V100 draw `df73ce66…` (Jean Zay 153094, receipt: Tesla V100-SXM2-32GB, torch 2.5.0, seed 20365833, recovery `c919329d…`); the stage refuses to run unless the tensor's raw-bytes hash is `df73ce66…`. No 3090 draw is used anywhere |
| Deployment | a new 12-MLP deployment at 3 % on 271001 (naive P&V, fault-aware pair rewrite, analog-LoRA r16 tiles of the 24 matrices), built locally (unit `deploy`), pinned in the same portable form; determinism check: two independent builds |
| Analog-LoRA B scale | same procedure as exp-001 (per-output-channel max \|B\| of the r16 digital LoRA at 3e-3 on this deployment; dead channels → 1e-12), trained on `df73ce66` (unit `bscale`) |
| Grids | exp-001's plus its edge extensions: digital LoRA r16 and r64 at {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}; analog LoRA r16 at {1e-4, …, 1e-2, 3e-2}; noisy, noiseless, Child-D (p = 0.1) at {1e-4, …, 1e-2} |

LoRA adapters (digital and analog) are on all 24 analog matrices. Recovery cohort `c919329d…`,
dev cohort `7411cf50…`, 300 steps × batch 8, cosine to zero, write laws, accounting, Fisher
estimator and seeds are unchanged.

## Declared implementation choices

1. Lifecycles: `make_lifecycles.py` units `l12-deploy`, `l12-bscale`, `l12-canary` (every screen arm,
   4 steps, development every 2), `l12-s1` (holds, noisy, noiseless; 12 rows), `l12-s2` (Child-D,
   digital LoRA r16/r64; 15), `l12-s3` (analog LoRA; 6). Pins in `pins.json["l12"]`. exp-001's
   lifecycles regenerate byte-identically.
2. Execution: `launch/run_local_l12.sh UNIT ROOT [TAG]` (public CLI: `ebl lifecycle plan`,
   `ebl campaign run`, `ebl runs inspect --verify-artifacts --require-complete`) from a clean
   clone of the pushed launch commit on `nom-cool-1`.
3. exp-001 deviations 1–3 (3090 draw, B scale on the 3090 draw) do not apply: the stored V100 draw is
   used throughout. exp-001 points 3–6 (portable pinned deployment made on the 3090, Child-D Fisher,
   on-chip cosine, final states) apply unchanged.

## Budget and execution

- Target: RTX 3090 `nom-cool-1`, conda `py312` (torch 2.5.1+cu121). One GPU, units sequential.
- Root: `nom-cool-1:/home/filip/analog-research-runs/fault-paper-tuning-l12/` (`inputs/`,
  `plans/<tag>/`, `runs/<tag>/`, `logs/`, `src-<commit>/`).
- Inputs (verified SHA-256): source `ed53b6cb…` (copied from Jean Zay
  `drift-all-mlps-20261111/inputs/source-l12-ed53b6cb.pt`), samples tensor `df73ce66…` (file
  `036a564c…`), teacher `2d74da66…` and corpora at `nom-cool-1:~/analog-research/opt-cmo-exp1/inputs/`.
- Expected ~100 s per training arm (2× exp-001), 31 training arms ≈ 1 h; storage < 20 GB.

## Pinned inputs

- **Deployment** `fault-paper-l12-deploy-271001-random3pct`, SHA-256
  `03af2f12921c6ded1dd4f033c29b5ba61f116bfd494cb03f7075255e66e8044d` (1.38 GB), from unit `deploy`
  at `dc9f3e9` (runs `deploy-a`, `deploy-b`). **Determinism:** the two independent builds wrote
  byte-identical files (same SHA-256) with identical apparent hashes — naive `0c6140f9…`,
  fault-aware `06bd5454…`, adapter `263bbf76…`; fault identity `4e01b48f…`, adapter fault identity
  `3132d761…`. Dev KL: clean 0.2041, naive 0.4362, fault-aware 0.2191. ~30 s per build, peak 6.2 GB.

## Current execution and monitoring handoff

Owner: the implementing agent. Next: unit `bscale`.
