# exp-001 (crossbar): learning-rate screen at 3 % random stuck devices, tuning chip 271001

State: implemented and canaried; Jean Zay jobs submitted for 22:00 on 13 November 2026
(handoff below). Branch: `exp/fault-paper-tuning`.

**Authorization (user, 13 November 2026):** the tuning runs of the shared frozen spec
(on-chip-learning `campaigns/fault-recovery-paper/series/001-lr-tuning/experiments/exp-001-lr-screen.md`,
issue #39) on the Jean Zay dev queues, starting at 22:00 today; the crossbar column only.
At most 5 crossbar jobs (the dev-QoS allowance is shared with the DRN agent). Local
canaries on the RTX 3090 `nom-cool-2` (shared; small runs, one at a time).

## Question and decision

Which learning rate does each crossbar recovery arm select at 3 % random stuck devices,
on tuning assignment 271001? The selected rates fix the recipes for the paper seeds
(271002–271004; not tonight). Rule (spec): final endpoint, lowest development teacher KL;
ties within 1 % → fewer writes. A selection at a grid edge is reported, not extended.

## Frozen comparison (spec values; all confirmed by the user's frozen spec)

| Setting | Value |
| --- | --- |
| Network | OPT-125M, analog fc1/fc2 of decoder layers 6–11 (28,311,552 weights), std3 whole-model HWA source `86522e23…b028`, FP32 teacher `2d74da66…74a9` |
| Device | CMO 2 % endpoint fit, G 9–88.2 µS, active/reference pair, per-output-channel abs-max scale, tile 256; no drift, no read noise |
| Faults | `random` (uniform 9–88.2 µS), permanent, per physical device, 3 % (30,000 ppm), assignment 271001; hierarchical placement with die and array concentration `null` (uniform rates, exact total per fault group, uniform devices within arrays) |
| Deployment | Gaussian-endpoint P&V (naive), then `fault_rewrite` verify/pair, 5σ, seed 241016 (fault-aware). All arms load the **same pinned portable deployment** (below) |
| Recovery data | WikiText-2 train, 1,024 × 256 tokens, block selection seed 20261104, cohort `c919329d…` (raw-bytes hash, as drift-regimes) |
| Target | 8 fixed teacher samples per position, the draw `df73ce66…` (seed 20261104 + 104729 = 20365833, batch 16, CUDA multinomial), loss = mean NLL of the 8 samples |
| Schedule | Adam (0.9, 0.999, 1e-8), 300 steps, batch 8 sequences drawn with replacement by `torch.randint` from one CPU generator seeded 20261104 (drift digital-arm order), cosine decay to zero (torch `CosineAnnealingLR(T=300)`), no warm-up |
| Development | WikiText-2 validation blocks 0–63 (`7411cf50…`), mean next-token KL against the full FP32 teacher, on the held apparent state, every 50 steps and at step 300 |
| Write law (noisy) | `ThresholdedWriteAdam`: signed accumulation of Adam commands, issued when |pending| > 2σ nominal mid-range (q units), one noisy CMO write per issued command on the active device, no cap |
| Write law (noiseless) | `IdealizedWriteAdam`, write noise 0: dense, one exact write per nonzero command and step |

Arms and grids (all from the same deployment; on-chip arms write active devices only):

| Arm | Implementation | Grid |
| --- | --- | --- |
| hold-naive | naive P&V state, evaluated | — |
| hold-fault-aware | fault-aware state, evaluated | — |
| noisy on-chip | thresholded noisy writes on every base weight | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 |
| noiseless on-chip | dense exact writes | same |
| Child-Tuning_D, p = 0.1 | Fisher = mean over the 128 consecutive batches of 8 recovery sequences of the squared gradient of the training loss w.r.t. the held normalized weights, at the fault-aware state, computed once per job; top 10 % per matrix → fixed mask; noisy thresholded writes inside the mask only | same |
| digital LoRA r16, r64 | `OptMlpSuffix.enable_lora` (α = r, A uniform ±fan_in^-1/2, B = 0, seed 271001) on all 12 analog matrices; physical state held, zero writes | 1e-3, 3e-3, 1e-2 |
| analog LoRA r16 | A, B of each matrix in extra CMO tiles (pairs, tile 256, names `<matrix>.lora_a/b`); their own fault groups/streams at 3 % and their own endpoint and rewrite streams; P&V + the same verify/pair rewrite at initialization (part of the pinned deployment); A target = the digital-LoRA initial A (same seed) / per-row abs-max, B target 0; B scale per output channel fixed from the pinned B-scale file; trained with the noisy thresholded law on adapter devices only; output added through the digital-LoRA path (α/r = 1) | 1e-4 … 1e-2 |

Accounting per write arm: healthy device writes, stuck write attempts, touched devices,
max writes per device, out-of-range healthy devices, writer diagnostics; Child-D also the
mask size/hash and writes outside the mask (must be 0). LoRA arms check that the base
deployment is bit-unchanged.

## Declared implementation choices and deviations (reported to the coordinator)

1. **Sample draw is GPU-specific.** `df73ce66…` was drawn on V100 (torch 2.5.0). The same
   procedure on the RTX 3090 gives `6bc43ffa…` (identical under torch 2.5.1 and 2.11), so
   the tensor cannot be produced locally and no stored copy existed. The screen units
   (`a1`–`a3`) load the V100 draw stored by the DRN half's draw job (Jean Zay 153094,
   `drn-fault-paper-lr-tuning-20261113/runs-9efad55/samples-v100/teacher-samples8-c919329d-df73ce66.pt`,
   written only if its hash is `df73ce66…`), run with `--dependency=afterok:153094`, and
   **refuse to train unless the loaded tensor's raw-bytes hash is `df73ce66…`**. The
   equivalent in-job V100 draw units (`j1`–`j3`) are kept for reruns. Local units
   (deploy, bscale, canary, probe, canaryfile) declare the 3090 draw.
2. **Analog-LoRA B scale (declared fallback).** From the r16 digital LoRA at 3e-3 run
   locally on the RTX 3090 on the pinned tuning deployment (unit `bscale`), which trains
   on the 3090 draw `6bc43ffa…` (point 1), not `df73ce66…`. Channels whose digital B row
   stayed exactly zero (dead fc1 units) get scale 1e-12 (no effective adapter there).
3. **Deployment made on the RTX 3090, pinned in portable form.** CMO endpoint draws depend
   on the GPU; to give every arm (and any GPU) the same deployed state the deployment is
   programmed once locally and loaded by hash (apparent-state hashes checked at load).
   Reproduce check: two independent local deployments gave identical apparent hashes.
4. **Child-D Fisher batches.** Batch-level squared gradients (8 sequences, as Child-Tuning's
   reference implementation accumulates per-batch grad²), gradient w.r.t. the normalized
   held weights q, top-10 % per logical matrix, no 1/p rescaling (Child-Tuning_D).
5. **On-chip cosine.** The writer's learning rate is set to the closed-form cosine before
   every step; digital LoRA uses `CosineAnnealingLR` (equal to 1e-6 relative, tested).
6. **Final states.** LoRA and analog-adapter states are saved per arm; base-array final
   states of the write arms are not retained (hash recorded) — tuning only.

## Readiness evidence (local RTX 3090, dirty probes, not evidence)

- Deployment: clean 0.04297, naive 0.18552, fault-aware 0.05475 dev KL; ~16 s.
- Canary (4 steps, every method): all complete; Child-D mask 2,831,160 weights, 0 writes outside.
- Probe (300 steps, 3e-3): noisy 0.0714 (25.8M writes), Child-D 0.0494, analog LoRA 0.0460;
  ~35–50 s per arm, peak 4.4 GB; digital LoRA r16 3e-3 (bscale) 0.0612.

## Budget and execution

- Target: Jean Zay V100-32G, `umg@v100`, `qos_gpu-dev`, 3 jobs, 1 h 30 limit each, held
  with `--begin=2026-11-13T22:00:00`. Expected 10–30 min each (3090: 35–50 s per arm).
  Cap 6 V100-hours including one retry per understood operational failure.
- Units: `j1` holds + noisy + noiseless (12 rows); `j2` Child-D + digital LoRA r16/r64 (11);
  `j3` analog LoRA (5). Storage < 10 GB.

## Current execution and monitoring handoff

(see the coordinator's issue #39 comments; filled at launch below)
