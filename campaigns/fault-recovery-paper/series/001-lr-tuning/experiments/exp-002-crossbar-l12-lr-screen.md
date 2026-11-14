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

- **Analog-LoRA B scale** `fault-paper-l12-bscale-r16-3em3`, SHA-256
  `e1aa6c11f21ebc3f0143b4815044d2e5c906c77cfe129a00480f931f2415c508`, from unit `bscale` at `9412191`
  (`launch/make_b_scale.py`; the r16 digital LoRA at 3e-3 on `03af2f12`, samples `df73ce66`; its
  final dev KL 0.1738, ~40 s). 24 matrices; per-matrix median 0.055–0.079, max 0.218; 2,116 fc1
  channels with B exactly zero (dead units) → 1e-12 at load.

## Readiness: canary (launch commit `ca1d470`, readiness only, not evidence)

Unit `canary`: all 33 screen arms × 4 steps, complete, `runs inspect --verify-artifacts
--require-complete` valid; samples `df73ce66…` and the three apparent hashes as pinned; clean
0.2041. Child-D mask 5,662,320 weights (10 % of 56.6M), 0 writes outside; Fisher 16 s. Noisy and
Child-D arms issue no writes within 4 steps (threshold gate, as expected); noiseless writes
~201.7M device writes per 4 steps; analog LoRA writes only at 3e-2 within 4 steps. Peak
allocated GPU memory 6.5 GB (nvidia-smi 8.2 GB) — well within the 24 GB 3090. ~3 min total.

## Collected screen (launch commit `ca1d470`; result interpretation belongs to the on-chip-learning record)

Units `s1`–`s3` complete, each `runs inspect --verify-artifacts --require-complete` valid; every
unit logs samples `df73ce66…` and apparent hashes naive `0c6140f9…`, fault-aware `06bd5454…`,
adapter `263bbf76…` (`launch/summarize_l12.py`, saved as `ROOT/logs/summary-l12.json`).
Wall time 11:57:53–12:22:53 (25 min); ~53 s per noisy/noiseless arm, ~55 s Child-D (+17 s Fisher),
~37 s digital LoRA, ~39 s analog LoRA. Peak allocated 6.5 GB (nvidia-smi 8.9 GB). Runs 5.9 GB.

Dev KL: clean 0.20412, naive hold 0.43616, fault-aware hold 0.21912. Final dev KL (bold = selected):

| Arm | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Noisy | 0.21912 | 0.18602 | **0.17641** | 0.19375 | 0.31046 | |
| Noiseless | 0.16250 | 0.15501 | **0.15576** | 0.17869 | 0.29287 | |
| Child-D p=0.1 | 0.21912 | 0.18651 | 0.17740 | **0.16942** | 0.18838 | |
| Digital LoRA r16 | 0.16765 | 0.16204 | **0.15967** | 0.17381 | 0.21601 | |
| Digital LoRA r64 | 0.16202 | **0.15802** | 0.16128 | 0.19362 | 4.01935 | |
| Analog LoRA r16 | 0.21916 | 0.21497 | 0.18655 | 0.16727 | **0.16119** | 0.16337 |

No selection is at a grid edge. Tie rule applied once: noiseless 3e-4 (0.15501) and 1e-3 (0.15576)
are within 1 %; 1e-3 has 225,675 fewer healthy writes (15,498,489,228 vs 15,498,714,903; dense
writes make the counts nearly equal), so 1e-3 is selected. Every trained arm except the zero-write
ones ends below the clean source KL (0.204): WikiText recovery training also adapts the source.

Write cost at the selected rates (300 steps):

| Arm | Healthy writes | Stuck attempts | Touched devices | Max/device | Out-of-range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noisy 1e-3 | 5,807,378 | 180,941 | 5,676,941 | 3 | 785,856 |
| Noiseless 1e-3 | 15,498,489,228 | 479,664,756 | 51,776,649 | 300 | 909,874 |
| Child-D 3e-3 | 6,100,002 | 189,777 | 3,963,569 | 10 | 754,185 (mask 5,662,320; 0 outside) |
| Analog LoRA 1e-2 (adapter tiles) | 10,775,207 | 333,124 | 1,355,667 | 31 | 69,443 (adapter) |
| Digital LoRA r16 1e-3, r64 3e-4 | 0 | 0 | 0 | 0 | 722,958 (base at deployment) |

## Current execution and monitoring handoff

Complete; nothing running on `nom-cool-1`. Interpretation and issue updates: the coordinator (#37, #39).

- **Screen:** units `s1` → `s2` → `s3` in one detached chain on `nom-cool-1`, ran
  2026-11-14 11:57:53–12:22:53 from `src-ca1d470` (clean clone of `ca1d470`), `launch/run_local_l12.sh UNIT ROOT`.
- **Logs:** `ROOT/logs/{s1,s2,s3}.log`, `screen.start`/`screen.end`, GPU sampler `gpu-mem.log`.
- **Runs:** `ROOT/runs/<unit>/opt125m-cmo-l12-fault-paper-tune-<unit>/stages/standard_hwa_model__271001/local/fault_tuning__standard_hwa_model__271001/attempt-001/runs/*/`.
- **Checks:** every unit inspected with `--verify-artifacts --require-complete`; `launch/summarize_l12.py ROOT`
  checks samples and apparent hashes in every unit and applies the selection rule.
