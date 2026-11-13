# Fault-recovery paper LR screen at 3 % faults — DRN half

## Question and assignment

Which learning rate does each DRN recovery arm need at 3 % random permanent
faults, on the tuning chip (assignment 271001)? The screen fixes the rates for the
paper seeds (271002–271004). The shared frozen specification is
[exp-001 of the fault-recovery paper campaign](../../../on-chip-learning/campaigns/fault-recovery-paper/series/001-lr-tuning/experiments/exp-001-lr-screen.md)
(issues #33, #34, #38, #39 on `filip137/on-chip-learning`). The user authorized
these runs on the Jean Zay dev queues, starting at 22:00 on 13 November 2026.
This note covers the DRN column; the crossbar half lives in `hwa/`.

Branch `drn/20261113-fault-paper-lr-tuning`, worktree
`/home/filip/analog-research/worktrees/drn-20261113-fault-paper-lr-tuning`.

## Frozen comparison (DRN column)

- **Network:** OPT-125M, six perfect-diode DRN MLPs (decoder layers 6–11), std3
  HWA source (contract `86522e23…`, mapped HWA input `dfb9f22f…`), FP32 teacher
  `2d74da66…`. Solver T = 4 settling + K = 4 gradient sweeps, evaluation 4 sweeps
  (the October fault contract).
- **Device and faults:** CMO 2 % endpoint fit, 9–88.2 µS, no drift, no read noise.
  Random stuck faults, uniform in [9, 88.2] µS, 3 %, assignment 271001, uniform
  placement.
- **Deployment:** the October fault-aware P&V (partner rewrite, 5σ). Every arm
  restarts from one saved deployment, including its device write RNG.
- **Recovery set:** WikiText-2 train, 1,024 × 256 tokens, cohort `c919329d…`
  (seed 20261104). Targets: the 8 fixed teacher samples per position, draw
  `df73ce66…` (seed 20365833); loss = mean of −log q over the 8 samples.
- **Schedule (all learning arms):** Adam β 0.9/0.999, ε 1e-8; 300 updates of
  batch 8; cosine decay to zero without warm-up (rate of update k is
  lr·½(1 + cos(π(k − 1)/300)), as torch `CosineAnnealingLR`); one shared batch order
  (seed 20261104).
- **Development and selection:** 64 WikiText-2 validation sequences (`7411cf50…`),
  mean next-token KL against the full FP32 teacher, at updates 0, 100, 200 and 300.
  The fixed final endpoint is the result; lowest KL wins, ties within 1 % go to
  fewer writes. No test sets.
- **Writes:** no cap. Noisy law and gate as in October: accumulate signed Adam
  commands and issue when |pending| ≥ 2ρ(G_target); targets clipped to
  [9, 88.2] µS before the noisy write; realized values are not clipped.

| Arm | DRN implementation | LR grid |
| --- | --- | --- |
| hold-naive | raw (pre-rewrite) programming of the same deployment | — |
| hold-fault-aware | the saved deployment | — |
| noisy | all four devices per weight, 2ρ-gated bounded Adam, write noise on | 1e-4 … 1e-2 (5) |
| noiseless | dense direct Adam on all devices, write noise 0 | 1e-4 … 1e-2 (5) |
| Child-Tuning_D, p = 0.1 | Fisher once at the deployment; top 10 % of devices per matrix (12 matrices); noisy law on masked devices only | 1e-4 … 1e-2 (5) |
| digital LoRA r16 / r64 | one adapter per DRN MLP: MLP input → MLP output, added to the DRN output; α = r; no writes | 1e-3, 3e-3, 1e-2 |

Analog LoRA is crossbar only (DRN deferred).

## Implementation

- Stage `fault_tuning` of `opt_cmo_drn_hwa.v1`
  ([runner](../../ebl/runtime/opt_cmo_drn/fault_tuning.py),
  [arms](../../methods/fault_tuning.py),
  [settings](../../methods/config/fault_tuning.py)). One run reads the deployment
  and runs its declared arms in turn. Each finished arm writes
  `artifacts/arm-<label>.json` and a `fault_tuning_arm` metric before the next starts.
  The config type is a `Config` subclass, so all earlier config digests are unchanged.
- Writers: `DirectWriteAdam` and `ThresholdedBoundedWriteAdam` take a
  `cosine_to_zero` schedule. Constant-LR contracts are unchanged.
- Reported per write arm: healthy writes (`device_writes`), stuck attempts,
  touched devices, maximum writes per device, devices outside [9, 88.2] µS, the
  bounded-target counters, per-array statistics and the dev-KL curve.
- Child-Tuning_D: the Fisher is the mean over the 128 consecutive batches of 8
  (cohort order, one pass) of the squared batch gradient per device, as in the
  Child-Tuning reference code. Every device counts as trainable, including stuck
  ones, because the controller cannot see them; the mask record logs how many
  selected devices are stuck. Masks are saved (`artifacts/child_d_masks.pt`).
- Digital LoRA: A ~ U(±1/√768) (seed 20261113 + block), B = 0, α = r (scale 1).
  It is trained with torch Adam and `CosineAnnealingLR` through the held physical
  view. The runner checks that the held state and write counters are unchanged.
  Adapters are saved.
- Configs: [`examples/opt_cmo_drn/fault_paper_tuning/`](../../examples/opt_cmo_drn/fault_paper_tuning/),
  written by [make_configs.py](20261113-fault-paper-lr-tuning/make_configs.py).
  Tests: `tests/test_opt_cmo_drn_fault_tuning.py`.

## Deviations and choices made in code (not scientific changes)

1. **Teacher samples were never stored.** `df73ce66` exists only as a logged hash
   (21 hwa runs, all V100 + PyTorch 2.5.0). The same procedure gives `6bc43ffa…`
   on the RTX 3090, both in hwa's own code and in this branch's
   [draw_samples.py](20261113-fault-paper-lr-tuning/draw_samples.py), so the CUDA
   multinomial stream depends on the GPU. Each V100 job therefore redraws the
   tensor with the identical procedure. It saves the tensor only if its hash is
   `df73ce66…`; otherwise the job exits with code 3 before any training. The local
   canary used the 3090 draw, labelled `canary-rtx3090.json`.
2. **Deployment made on the RTX 3090.** Two deploy runs were bit-identical (file
   SHA-256 `159848f7…8d8cca`). The file was copied to Jean Zay and its hash checked
   there. A V100 deployment could be a different realization of the same
   assignment; all DRN arms share this one.
3. **Forward policy:** the October fault contract (8 sweeps with gradients through
   the last 4 in training, 4 settling sweeps in evaluation). It is not the
   drift-regimes `eval8` policy.
4. **Batch order:** sampled with replacement, `torch.randint` with seed 20261104,
   one order shared by all arms. This is the drift digital-arm convention for
   arm index 0.
5. **LoRA size:** the spec's single adapter per DRN MLP has 6·r·1,536 parameters:
   147,456 at r16 and 589,824 at r64. The crossbar puts adapters on fc1 and fc2,
   so the two platforms' parameter counts differ.
6. **Microbatch 1** for training, Fisher and evaluation (the October convention).

## Execution

- Local readiness on the RTX 3090 (nom-cool-2):
  - 10 new tests and the 180 OPT CMO DRN tests pass.
  - Deterministic deployment: see deviation 2.
  - Canary `results/fault-paper-lr-tuning/local/canary-1/run`: all six methods at
    3 updates; native bundle complete and valid.
- Jean Zay. The V100 dev queue was full (10/10 submitted jobs, other studies)
  when this setup was ready, so the launch is mixed. All jobs carry
  `--begin=2026-11-13T22:00:00`.
  - One V100 job ([draw.slurm](20261113-fault-paper-lr-tuning/launch/draw.slurm),
    `umg@v100`, `qos_gpu-dev`, `v100-32g`, 20 min) reproduces the samples. It
    writes them to `runs-<commit>/samples-v100/` only if the hash is `df73ce66…`.
  - Five A100 dev jobs (`umg@a100`, `gpu_p5`, `qos_gpu_a100-dev`, `-C a100`, 8 CPUs,
    2 h, `--dependency=afterok:<draw>`) each run one packed config:
    - A: holds + noisy ×5;
    - B: noiseless ×5;
    - C: Child-D (Fisher + ×5);
    - D: LoRA r16 ×3;
    - E: LoRA r64 ×3.
  - If the draw fails, no training job starts.
  - Submitted by [jz_submit_hybrid.sh](20261113-fault-paper-lr-tuning/launch/jz_submit_hybrid.sh)
    as soon as one V100 dev slot was free. The all-V100 variant
    ([jz_submit.sh](20261113-fault-paper-lr-tuning/launch/jz_submit.sh); each job
    draws its own samples) is kept for reruns.
- Inputs on Jean Zay (`$SCRATCH` = `/lustre/fsn1/projects/rech/umg/ucy17uy`):
  - teacher `opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin`;
  - HWA `opt125m-cmo-drn-rates-v1/inputs/hwa-std3.pt` (`dfb9f22f…`, checked);
  - WikiText `opt-cmo-exp1/inputs/corpora/wikitext-2-raw-v1.pt` (`33e95834…`, checked);
  - deployment `drn-fault-paper-lr-tuning-20261113/inputs/deployed-3pct-271001.pt`.
- Source: `git archive` of the launch commit, staged read-only at
  `$SCRATCH/drn-fault-paper-lr-tuning-20261113/source-<commit>/` with
  `source_snapshot.json`. On the login node with PyTorch 2.5.0, 56 tests pass,
  including the 10 new ones. The legacy `ThresholdedWriteAdam` golden fails, as
  documented in October.
- Outputs: `$SCRATCH/drn-fault-paper-lr-tuning-20261113/runs-<commit>/<config>/run/`,
  Slurm logs in `runs-<commit>/slurm/`.
- Retention: no final banks are saved (held hashes are recorded); masks and
  adapters are kept. New storage is under 100 MB.
- Expected wall time per job, measured on the 3090 at microbatch 1:
  - 1.13 s/update for write arms, 0.75 s/update for LoRA;
  - Fisher pass 195 s;
  - development evaluation 2.5–6 s.
  The A100 should be at least as fast. On the 3090 scale, A, B and C take about
  35 min each and D and E about 15 min each. A V100 at 2 s/update would need about
  60 min for each of A, B and C. All fit the 6,900 s run budget.
- Collection: `python campaigns/pilots/20261113-fault-paper-lr-tuning/collect.py <runs dir>`
  tabulates arms and applies the selection rule (lowest final dev KL; within 1 %
  the fewest write attempts; grid edges flagged).

## Current handoff

Filled in at launch (commit, staging path, job IDs).
