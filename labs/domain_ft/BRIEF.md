# Domain fine-tuning: preliminary runs (exp-001), night of 2026-11-07 → 08

Written by the main session (planner, effort max) for the implementing and
monitoring agent (effort high). The main session turns your results into the
campaign plan by 08:00. Everything here is **exploratory lab-harness evidence**,
not `ebl` lifecycle evidence.

## 0. Goal and deliverables (due 06:45 CET, Sun 8 Nov 2026)

1. **Dataset gate:** digital full fine-tuning (FT) gains for 8 candidate targets.
2. **Learning-rate (LR) screens:** 3-point grids for 8 PEFT methods on MNLI, HellaSwag and
   PubMed text (then RCT if time allows).
3. **Capacity sweep:** adapter size vs quality for continued pretraining on PubMed
   and Python code.
4. **ΔW statistics:** compare fine-tuning weight changes with the CMO write resolution.
5. **If feasible:** a deployed-crossbar domain-gap measurement and an LR-transfer
   check on a deployed chip replica (Tier D).
6. **Timing:** seconds per update and peak memory for every run type.

Plus `summary/` tables (section 9), `MONITOR.md`, and a final report.

## 1. Hosts, environment, etiquette

- **GPUs: riri and trex only** (one RTX 5090 32 GB each, both idle at 22:10).
  loulou runs another user's job; fifi is unreachable. Never touch other users'
  processes. If a host becomes unavailable, move its queue to the other one and
  tell main.
- **Python on both hosts:** `/home/filip/miniconda3/envs/py312/bin/python` (3.12.13,
  torch 2.11.0+cu128, transformers 5.8.0, datasets 2.21.0, pyarrow,
  pandas, huggingface_hub 1.14). `peft` and `accelerate` are absent; implement the
  methods yourself (they are small).
- **Network:** both hosts and nom-cool-2 reach huggingface.co. Pin every dataset by
  repo + revision (commit hash) + file SHA-256.
- **Disk:** trex `/home` is 91% full (171 GB free). Keep our footprint under
  20 GB per host. **Never save full-model checkpoints**; compute ΔW statistics
  in-process. Small PEFT parameter files (≤ a few MB) may be saved.
- **Remote jobs:** use `ssh -o BatchMode=yes`. Run them in tmux sessions named `dft-*`,
  and always address them as `-t =name`, because a bare `has-session` prefix-matches.
  Background `ssh … &` can hang, so run prepare/staging steps in the foreground.
- **Local GPU:** do not use nom-cool-2's 3090; the agent sandbox blocks it.
- **Run roots on hosts:** `~/domain-ft-20261108/` (code snapshot, data, runs).
- **Local collection root:**
  `/home/filip/analog-research/on-chip-learning/artifacts/domain-finetuning/prelim-20261108/`
  (git-ignored). Sync results here at every check.

## 2. Code location and reuse

- **Your worktree:** `/home/filip/analog-research/worktrees/hwa-domain-ft`, branch
  `exp/domain-finetuning` from hwa main `e70a776`. It was created for you.
  - Put lab code under `labs/domain_ft/`.
  - Commit locally at milestones. **Never push**: origin is a public GitHub repo.
  - Do not modify `/home/filip/analog-research/hwa`, `/drn`, other worktrees,
    or the on-chip-learning campaign records. Main owns the records; you write
    only to the artifacts root above.
- **Task scoring semantics:** reuse `networks/tasks.py` in your worktree:
  `encode_pair`, `continuation_log_probs`, `row_sums`, `decision_scores`,
  `choice_loss`, `paired_standard_error`, lm-eval v0.4 templates, and BOS (id 2)
  at the start of every scored row.
  - **Preferred:** extend its `TASKS` registry *on this branch* with the new tasks.
    The native lifecycle can then reuse the same task files later.
  - Keep existing behaviour bit-identical: the existing hellaswag/sst2/piqa
    files must still validate.
- **Existing data** (git-ignored; copy from the main checkout):
  - `/home/filip/analog-research/hwa/data/corpora/{hellaswag,sst2,piqa}-opt125m-v1.pt`
    and `-stream.pt`; receipts are in `/home/filip/analog-research/hwa/data/tasks/`.
  - `/home/filip/analog-research/hwa/data/corpora/wikitext-2-raw-v1.pt`
    (schema `crossbar_lifecycle.token_corpus.v1`; `splits` train/validation/test
    are 1-D token tensors with one BOS at the start of each split).
  - `/home/filip/analog-research/hwa/data/corpora/opt125m-synthetic-50m-synthtest.pt`:
    the HWA training distribution's test text, used by Tier D1.
- **Model:** `facebook/opt-125m`, pinned revision
  `27dcfa74d334bc871f3234de431e71c6eeba5dd6` (as used by hwa).
  - Load in FP32 (`dtype=torch.float32`; transformers 5 renamed `torch_dtype`).
  - The hosts also hold the pinned teacher at `~/explore-20261101/inputs/teacher/`.
    Use either source, but record which, and record the weight file's SHA-256.
- **Deployed crossbar replica (Tier D)**
  - The std3 crossbar source is staged on both hosts at
    `~/explore-20261101/inputs/crossbar-std3.pt`. Verify and record its SHA-256.
    hwa's std3 source is `86522e23…`; if that receipt sits next to the file, check it.
  - Loading example:
    `/home/filip/analog-research/on-chip-learning/explorations/fisher-viability-20261102/hwa/labs/explore_b/refresh_lab.py`,
    function `build()`, which calls
    `opt_mlp.OptMlpSuffix(teacher, 6, 256, 'per_output_channel_absmax', 'model')`
    and loads the source. Note that it lives in an older code layout (a separate
    lab clone).
  - Device law: `/home/filip/analog-research/hwa/docs/reram_cmo_device.md`,
    section "Encoding":
    - q ∈ [−1, 1] per output channel, scaled by that channel's absolute maximum;
    - active device G_a* = 48.6 + 39.6·q µS; reference device G_r* = 48.6 µS;
    - read q̂ = (G_a − G_r)/39.6;
    - 2% acceptance: σ_PV(G) = 0.0112185391 + 0.01129027418·G µS;
    - clamp each programmed draw into [9, 88.2] µS;
    - no faults, no drift, no read noise.

## 3. Common training settings (all runs unless stated)

- **Precision:** FP32 parameters; allow TF32 matmuls (`torch.backends.cuda.matmul.allow_tf32 =
  True`, same for cudnn); no autocast. Dropout off (`dropout = attention_dropout = 0`).
- **Sequences:** max 256 tokens. For tasks, truncate the context from the left
  if needed; the continuation stays whole.
- **Optimizer:** AdamW, β (0.9, 0.999), ε 1e-8, weight decay 0. Clip the global grad norm at 1.0.
- **Schedule:** linear warm-up over 6% of updates, then cosine decay to 0.
- **Batch:** 32 examples per update. A task example carries all K choices; an LM
  example is one 256-token block. Use microbatching/accumulation freely to fit
  memory, keeping the effective batch at 32.
- **Budget:** **U = 1,000 updates** (or 4 epochs, whichever is first; this matters
  only for BoolQ). This is the proposed campaign default; DRN updates are expensive.
- **Seeds:**
  - data order 20261108, the same for every run on a dataset, so batches are paired across methods;
  - adapter init 41;
  - fixed development subsets (seed 20261108).
- **Dev evaluation:** at 250/500/750/1000 updates. **Test evaluation:** at the end only.
- **Memory tip:** for choice tasks, apply `lm_head` only at the scored continuation
  positions. Full logits over 32×K×256 positions × 50,272 vocabulary entries do
  not fit.

### Objectives and metrics

- **Choice tasks**
  - Loss: cross entropy over the K choices of the **per-token mean**
    continuation log-likelihood (`token_mean`, the exp-018 decision).
  - Selection metric: dev **acc_norm** (lm-eval character-length normalization,
    as `decision_scores`). Also report acc and choice CE.
- **LM (domain text)**
  - Loss: next-token CE over the 255 predicted positions of each block.
  - Metric: token-weighted perplexity on dev/test blocks.
- **Retention**, computed at the end of every run; never used for selection:
  - WikiText-2 test: perplexity, plus teacher KL(OPT-FP32 ‖ model) per token over 128 blocks;
  - PIQA test acc_norm (hwa task file);
  - ARC-Easy test acc_norm;
  - LAMBADA (OpenAI) last-word greedy accuracy, 2,000-example fixed subset.

## 4. Datasets

Candidate HF sources are listed below. Verify each one. If a source is
unavailable or script-only, pick an equivalent pinned parquet source and document it.
Record counts, token counts and truncation rates in a receipt JSON per dataset.

| Key | Source (candidate) | Template (context → continuations) | Train / dev / test |
|---|---|---|---|
| `mnli` | `nyu-mll/glue`, config `mnli` | `"{premise}\nQuestion: {hypothesis} True, False or Neither?\nAnswer:"` → `" True"`, `" Neither"`, `" False"` (labels 0/1/2) | train minus dev; dev = 2,000 from train, grouped by premise; test = validation_matched (report) **and** validation_mismatched (report separately) |
| `rct` | `armanc/pubmed-rct20k` (or another PubMed 20k RCT parquet) | `"Sentence from a clinical trial abstract: {text}\nSection:"` → `" background"`, `" objective"`, `" methods"`, `" results"`, `" conclusions"` | official train; dev = 2,000 sentences from official dev, grouped by abstract; test = official test (~30k) |
| `hellaswag` | existing hwa task file | as built | 37,905 / 2,000 / 10,042 |
| `boolq` | `google/boolq` | `"{passage}\nQuestion: {question}?\nAnswer:"` → `" no"`, `" yes"`. Truncate the **passage tail** to fit 256 tokens; record the rate | train minus dev; dev = 1,000 from train, grouped by passage; test = validation (3,270) |
| `casehold` | `casehold/casehold` (or `coastalcph/lex_glue` case_hold parquet) | context = `citing_prompt.split("<HOLDING>")[0]` (left-truncated to fit); continuations = the 5 holdings, appended directly (the context ends with "(") | official train; dev = 2,000 from official validation; test = official test |
| `sst2` | existing hwa task file | as built | as built (test 872) |
| `pubmed` | `ccdv/pubmed-summarization`, field `abstract` (strip sentence markers if any) | LM stream: BOS + abstracts joined by id 2, cut into 256-token blocks | blocks from train; dev 128 blocks from validation; test 256 blocks from test. Drop train abstracts that contain any RCT **test** sentence of ≥ 10 words (exact match after whitespace normalization); record the count |
| `code` | `codeparrot/codeparrot-clean-valid` or another permissive Python-file parquet with repo names | LM stream as above (files joined by id 2) | split **by repository** into train/dev/test; ≥ 10M train tokens; dev 128 and test 256 blocks |
| probes | `allenai/ai2_arc` (ARC-Easy test 2,376); `EleutherAI/lambada_openai` (test) | ARC: `"Question: {question}\nAnswer:"` → `" {choice}"`; LAMBADA: context → last word | probes only |

**Majority-class baseline:** the dev label distribution (for MNLI/RCT/BoolQ),
and chance for balanced multiple choice. LM corpora use the token_corpus.v1
schema, so the native pipeline can reuse them.

## 5. Methods (layers are zero-based 6–11 = decoder blocks 7–12, the analog MLP blocks)

Each method's trainable parameter count is listed; assert it in a unit test.
Adapters must be **output-neutral at init**: unit-test that logits are
identical to the base model's.

| Key | Trainable parameters | Count | LR grid |
|---|---|---:|---|
| `zs` | none (evaluation only) | 0 | — |
| `full` | all OPT-125M parameters (tied embeddings and head included) | 125.2M | 1e-5, 3e-5, 1e-4 |
| `mlp` | `fc1.weight`, `fc2.weight` in layers 6–11: the ideal-analog ceiling (perfect writes) | 28,311,552 | 1e-5, 3e-5, 1e-4 |
| `digital` | the plan's digital_only scope in layers 6–11: q/k/v/out weights and biases, both LayerNorms, fc1/fc2 biases | 14,215,680 | 3e-5, 1e-4, 3e-4 |
| `ln` | both LayerNorm affines in layers 6–11 | 18,432 | 1e-3, 3e-3, 1e-2 |
| `bitfit` | all biases in layers 6–11 (q,k,v,out, fc1, fc2, both LN biases) | 50,688 | 3e-4, 1e-3, 3e-3 |
| `ia3` | per-channel scale (init 1) on the MLP hidden activation after ReLU, before fc2 | 18,432 | 1e-3, 3e-3, 1e-2 |
| `lora_fc16` | LoRA on fc1 and fc2, r 16, α 32 (scale 2); A ~ U(±1/√in), B = 0 (hwa convention) | 737,280 | 3e-4, 1e-3, 3e-3 |
| `padapter64` | parallel adapter around the MLP. Input = `final_layer_norm` output (the MLP input); Δ = 4·W_up·ReLU(W_down h + b_down) + b_up, added to the MLP output before the residual add. W_down 768→64 ~ U(±1/√768); W_up = 0, biases 0 | 594,816 | 1e-4, 3e-4, 1e-3 |
| `lora_attn16` | LoRA on q_proj and v_proj, r 16, α 32 | 294,912 | 3e-4, 1e-3, 3e-3 |

**Capacity-sweep sizes:** `lora_fc{4,64,256}` (α = 2r) and `padapter{16,256,1024}`.

**Edge rule:** if a (dataset, method) selection lands on a grid edge, queue
**one** extension LR at the next 1-3-10 step beyond it (lower priority). It joins the selection.

## 6. Selection rule (dev only; test is never read by selection code)

- Select the LR with the best **final** (U = 1,000) dev metric.
- Ties go to the lower LR. A tie is within 0.2 pt for tasks, or within 0.5% relative perplexity for LM.
- Also record the best-over-checkpoints value and the full dev curve. Flag `edge`
  when the selected LR is at a grid boundary.
- Write `test.json` separately; selection reads only dev.

## 7. Tiers, in priority order (each GPU pulls from one shared ordered queue)

- **P0. Readiness, before 00:00 if possible.**
  - Unit tests: parameter counts, init neutrality, and scorer equality with
    hwa on 20 examples.
  - A smoke run per method (20 updates).
  - Timing probe: seconds per update for full / lora_fc16 / ln on MNLI, HellaSwag and pubmed.
  - Test whether **2 concurrent runs per GPU** give ≥ 1.3× aggregate throughput.
    If they do, run 2 per GPU.
- **P1. Tier A, the gate.**
  - `zs` on every task, the domains and the probes.
  - `full` × 3 LRs on mnli, hellaswag, rct, boolq, casehold, sst2, pubmed and code (24 runs).
- **P2. Tier B, the screens.**
  - The 8 non-full methods × 3 LRs on **mnli, hellaswag, pubmed** (72 runs).
  - Order by method (lora_fc16, padapter64, ln, bitfit, mlp, digital, ia3,
    lora_attn16), interleaving the three datasets, so every dataset has the
    key methods first.
- **P3. Tier D1, domain gap (evaluation only; insert as soon as implemented).**
  - Corpora: synthetic test, WikiText-2 test, pubmed test, code test (256 blocks each).
  - Models:
    - OPT-FP32;
    - std3 source **clean**: its trained digital parameters plus exact analog weights;
    - std3 **deployed**: 3 independent CMO programming draws (lab seeds 1–3).
  - Metrics: perplexity, and teacher KL against OPT-FP32.
  - Output: per corpus, deployed − clean (the analog-specific gap) and clean − OPT
    (the HWA gap).
- **P4. Tier C, capacity.**
  - code: `lora_fc16`, `padapter64` and `mlp` × 3 LRs (9 runs).
  - Then on pubmed **and** code: `lora_fc{4,64,256}` and `padapter{16,256,1024}`, each at the
    selected LR of its r16/w64 sibling for that domain. For r256 and w1024, also run one step lower.
  - `full`, `mlp` and `digital` come from tiers A/B/C.
- **P5. Tier D2, LR transfer on the deployed replica (draw 1).**
  - `lora_fc16`, `padapter64`, `ln`, `bitfit` × 3 LRs on hellaswag (12 runs).
  - Plus `lora_fc16` at its selected LR on the clean std3 source (1 run).
  - Compare optimal LRs and retained gain with Tier B.
- **P6. Extensions.** Edge-rule LRs; RCT screens (24 runs); `lora_fc16` and `padapter64` × 3 LRs on
  boolq and casehold; then two 2,000-update budget-sensitivity runs (`lora_fc16`
  at its selected LR on mnli and pubmed).

**Time rules:**
- Stop launching new runs at **06:15**. Let running jobs finish only if they end by 06:35; otherwise kill them and mark them `killed_deadline`.
- If you fall behind, drop from the bottom of this list. Document every drop.

## 8. ΔW statistics (in-process at the end of `full`, `mlp` and every `lora_fc*` run)

For fc1/fc2 of layers 6–11, with W0 = OPT weights and ΔW = W_final − W0
(merged (α/r)·B·A for LoRA), compute per matrix:

- `rel_norm` = ‖ΔW‖_F / ‖W0‖_F. Singular values of ΔW; report the
  rank capturing 50/90/99% of ‖ΔW‖² (`r50`/`r90`/`r99`).
- **Crossbar units** (per-output-channel scale s_j = max_i |W0_ji|):
  - Δq = ΔW/s_j and q0 = W0/s_j;
  - write noise σ_w(q0) = σ_PV(48.6 + 39.6·q0)/39.6 (active device only);
    the threshold uses σ_mid = σ_PV(48.6)/39.6 ≈ 0.01414;
  - report `rms_dq_over_sigma_mid`, `frac_abs_dq_gt_2sigma_mid` (the thresholded
    writer's 2σ rule), `frac_abs_dq_gt_sigma_w`, and `energy_frac_gt_2sigma_mid`
    (share of ‖Δq‖² carried by entries above 2σ_mid).
- **DRN units** (one scale per matrix: S = max |W0|, ŵ = W/S; the active-sign
  device sits at G = 9 + 79.2·|ŵ|, and ΔG = 79.2·Δŵ):
  - the same three statistics with σ_DRN(ŵ0) = σ_PV(9 + 79.2·|ŵ0|)/79.2;
  - the threshold uses σ_PV at the matrix's median G.

Write `deltaw.json` per run.

## 9. Outputs

- **Per run:** `runs/<dataset>/<method>__lr<lr>/` containing:
  - `config.json` (full spec, code commit, host, GPU, versions, data receipts' hashes);
  - `metrics.jsonl` (train loss every 10 updates; dev evaluations);
  - `final.json`: dev final/best, params, seconds per update, peak memory, wall time, status;
  - `test.json`, `retention.json`, and `deltaw.json` where applicable;
  - small PEFT parameters (`peft.pt`) for methods ≤ 1M parameters.
- **Run status values:** `done`, `failed` (with traceback), `killed_deadline`.
- **`summary/` (from `aggregate.py`; re-run at every check):**
  - `gate.csv`: dataset, zs dev/test, majority, best full LR, full dev/test, dev gain vs max(zs, majority), n_test, pass.
    The gate rule is **dev gain ≥ 10 pt and n_test ≥ 3,000**; SST-2 is a control.
  - `screen.csv`: every run row.
  - `selection.json`: per (dataset, method), the selected LR, edge flag, and dev/test at that LR.
  - `capacity.csv`: domain, method, size, params, LR, dev/test perplexity, WikiText perplexity and KL.
  - `deltaw.csv`; `domain_gap.csv`; `transfer.csv`; `timing.csv`.
- **Receipts:** `data/receipts/*.json`.
- **`MONITOR.md`:** timestamped events, failures, decisions and drops.
- **`status.json`:** queue counts per tier and host, ETA, last check time.

## 10. Monitoring and reporting

- **Check every 20–30 min.**
  - Wait with a background Bash command (`run_in_background: true`) whose
    `until` loop exits at the next check time, or earlier when a queue drains or a
    runner dies. Do not busy-poll in the foreground.
  - At each check: sync results, re-run `aggregate.py`, update `MONITOR.md` and
    `status.json`, and fix or relaunch failures (at most one retry per run;
    retries are not replicates).
- **SendMessage to `main`:**
  - (a) when P0 is done and the queue is launched: timing, concurrency
    decision, deviations;
  - (b) when Tier A is complete: the gate table;
  - (c) on any blocking problem;
  - (d) at the end.
- Keep messages short; point to files.
- **Do not end your task before the final report (~06:45)** unless main tells you to stop.
  If you end early by mistake, main will message you to continue.

## 11. Decision authority

- **You may:**
  - change microbatching or accumulation, per-GPU concurrency, and evaluation
    batching;
  - reorder runs within a tier;
  - drop lowest-priority items when behind schedule;
  - fix bugs.

  Document every such change.
- **Ask main first** (SendMessage) before changing templates, objectives, LR grids,
  budgets, method definitions or tier priorities.
  - Exception: if a definition is impossible as written, implement the closest
    version, document it, and tell main.
- **Escalate** if:
  - a dataset cannot be obtained;
  - a full-FT result falls below zero-shot on dev;
  - throughput is below 50% of the P0 estimate;
  - a host disappears.
