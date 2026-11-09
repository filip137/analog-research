# Sparse equilibrium adaptation first pilots

User authorization: 9 November 2026, test the Explorer proposal, first runs at
23:00 Europe/Paris on Jean Zay V100 dev queue. Earliest start is
2026-11-09T23:00:00+01:00. This is exploratory simulation evidence.

Question: do sparse conductance updates provide useful task adaptation beyond
matched sparse explicit updates, and does the difference depend on nonlinear
constraints? The shared campaign and task exploration are in
`on-chip-learning/campaigns/sparse-equilibrium-adaptation/`.

## Frozen first screen

One seed (271109), ideal writes, no faults, drift or read noise. Report ideal
device assignment commands and unique cells; no pulse or measured energy claims.

1. Small graph: 4x4 grid, grounded quadratic energy, linear and nonnegative
   variants. A target domain changes four edge conductances. Paired input cohorts
   of 512 train/128 development/256 test. Compare frozen, random/Fisher/known-edge
   sparse graph updates (K=4), full graph and explicit MLP controls after source
   fitting. Source fitting and adaptation are separate. This circuit-generated
   target is a mechanistic positive control, not broad superiority evidence.
2. OPT-125M: existing crossbar std3 and DRN std3 sources; adapt MLP indices 6–11.
   Counterfactually augmented sentiment at repository commit
   `6f232a1d2a11462a30ce08fb4825b734ab30828e`, paired original/revised data.
   Fit source on originals for 128 updates, then clone that state for adaptation.
   Adapt 128 updates at effective batch 8, context cap 128 tokens, microbatch 1.
   Use training pairs only; development pairs select among rates 3e-4 and 1e-3.
   Test stays unopened by the training worker in this first development screen.
   Full analog, random sparse and empirical-Fisher sparse use normalized physical
   device coordinates and Adam; K=46,080, fixed equal quotas over 12 matrices.
   Fisher uses squared individual-example gradients on 16 fixed training rows.
   Digital controls: input/output boundary affine, scoped MLP bias-only and
   rank-5 parallel linear LoRA. Each gets the same two-rate allowance. LoRA has
   46,080 scalars; boundary/bias counts are reported at their actual sizes.
   Frozen attention, embeddings, norms and LM head throughout. Binary label-token
   logit scoring uses the same prompt and positive/negative tokens for both models.

The OPT source-fit rate is 3e-4 in normalized conductance coordinates. This is
a first rate screen, not a claim of converged capacity. The short context may
truncate a causal edit: record identical-token pairs and exclude them from paired
adaptation/evaluation, preserving exclusions in the data receipt. Keep standard
dataset split assignments and check original/revision grouping and duplicates.
Original identity uses exact text after whitespace normalization. Pairs without
one unambiguous original are excluded with an explicit reason; no fuzzy matching.
Development pairs overlapping training text after whitespace normalization are
excluded as whole pairs; retain their IDs and reasons in the data receipt.

DRN forward uses 24 total sweeps in both training and evaluation (16 settling
plus 8 differentiated in training). Canary compares against longer relaxation
and a longer gradient tail; failures block the batch, without silent retuning.
Sparse physical writes affect only declared devices; crossbar references remain
fixed. Source fits and each method's starting metrics are retained separately.

## Decisions and budgets

Primary graph endpoint: held-out MSE vs ideal write commands, with active-set
and response-rank diagnostics. Primary OPT endpoint: development original,
revised and both-correct pair accuracy, alongside source-fit headroom and writes.
Null/negative comparisons are valid. Single-seed development results do not
support a confirmed architecture advantage.

One same-wrapper canary (array 0-0, <=20 min), then array 0-1%2: task 0 runs
graph + MLP sentiment; task 1 runs DRN sentiment. Each full allocation is bounded
by 2 h on one V100-32GB, 10 CPUs, account umg@v100, partition gpu_p13,
qos_gpu-dev; PyTorch module pytorch-gpu/py3/2.5.0. Maximum allocated allowance
4 h 20 min. Source/data/results combined <=12 GB incremental storage.
Stop at the wall-time guard and preserve completed cases as partial coverage;
no automatic extra allocation. A failed canary prevents full submission.
Require completion by 06:00 Paris on 10 November; Slurm's deadline bounds late
starts. The worker stops after 6,600 seconds (canary 1,050), leaving teardown
time inside the allocation. Memory request is 80 GB host RAM and V100-32GB.

Data freeze: 1,630 train and 231 development pairs. Training exclusions: four
ambiguous original identities and 73 identical truncated prompts. Development:
one ambiguous identity, 12 identical prompts, one training-text overlap. The
prepared data preserves original/revised labels, token IDs, pair IDs, text hashes,
truncation flags, excluded IDs/reasons and pinned source-file hashes.

## Handoff

Branch: drn/sparse-equilibrium-pilot. Numerical work uses the public
`python -m ebl train --config <config.json> --output-dir <destination>` and native RunStore
bundles. The exact staged commit, commands, tests, Slurm IDs and durable watch
are filled before scheduling. Scheduling/resume must never duplicate a parent.
The watcher owns gating, observation, collection and the terminal review handoff.

Implementation: `experiments/sparse_equilibrium/launch.py` reuses the native Slurm
accounting and locking helpers. Launch configs are in `examples/sparse_equilibrium/`.
The same `recovery.slurm` wrapper handles the explicit canary mode and production.
`launch/manifest.json` in the staged source records both exact commands, all
effective source/config hashes and the passing static receipt. The release command
checks successful singleton accounting, native artifact hashes, coverage and logs,
then submits one array. A persisted uncertain submission blocks duplicate attempts.

Static checks: `bash -n experiments/sparse_equilibrium/recovery.slurm`, compilation
of changed Python, and the sparse-equilibrium, architecture-boundary, public-CLI
and native Slurm tests. Source and configs are immutable after the live canary.
