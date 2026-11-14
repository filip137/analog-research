# OPT-125M with all 12 MLPs as DRNs: std3 HWA source

Requested by the user on 2026-11-14 for the fault-recovery paper
([on-chip-learning #37](https://github.com/filip137/on-chip-learning/issues/37)):
repeat the 6-MLP DRN HWA recipe with all 12 decoder MLPs as DRNs, on one RTX 5090.

**Question.** Does the 6-MLP std3 DRN HWA recipe, unchanged, train a usable 12-MLP
DRN source for the paper fault sweep? This run produces that source; the sweep itself
is planned in on-chip-learning `campaigns/fault-recovery-paper/`.

## Frozen recipe

Copied from the std3 five-pass arm of
[20261016-opt125m-cmo-drn-hwa-first](20261016-opt125m-cmo-drn-hwa-first.md)
(config `plans/execution/full-std3.json` on loulou, resolved SHA-256 `d5e5c2b9…`).
Only two fields change:

| Field | 6 MLPs | 12 MLPs |
| --- | --- | --- |
| `analog_layers` | 6 (layers 6–11) | 12 (layers 0–11) |
| `source_sha256` | `86522e23…` (6-MLP crossbar std3) | `ed53b6cb…` (12-MLP crossbar std3 recipe, hwa `exp/drift-all-mlps`, epoch 98) |

Unchanged: perfect-diode 1536–6144–1536 DRN per MLP, `full_span_quad.v1` mapping,
T=4, K=4, A=5, B=0.1; std3 HWA, recipe B (Adam 0.9/0.98, eps 1e-6, clip 1),
conductance and calibration LR 1e-4, digital LR 3e-6, 245-update warm-up then cosine to
20 %; five passes of the 195,226-sequence synthetic corpus (30,505 updates, batch 32 × 256);
selection every 512 updates on development teacher KL (256 sequences) after programming and
partner rewrite, nominal and random 3 %, assignments 211001/211002; retention
`best_latest_final`. No optimizer re-tuning.

Code: the schema accepted only six DRN MLPs; it now accepts 6 or 12
(`ebl/schema/opt_cmo_drn.py`). The runtime was already parameterized by depth.
`reconstruct_source` rejects a source whose analog weights do not match the depth.
Tests: `tests/test_opt_cmo_drn.py` (12-MLP schema, reconstruction, gradients to all 12 blocks).

Configs: `examples/opt_cmo_drn/l12/{prepare,hwa_readiness,hwa_std3}.json`.

## Expected outcome and reference

6-MLP std3: 1.77 s/update, 55,853 s total (15.5 h) on loulou; peak 5 GiB; selected at
the final step, development KL 0.0720 (nominal 0.0229, random 3 % 0.1211).
With twice the DRN blocks we expect about 3.5 s/update and about 31 h. The 12-MLP crossbar
source starts worse (nominal KL 0.073 against 0.021 at 6 MLPs), so a higher final KL than
6 MLPs is expected. If the selected KL is far worse, the source may need a longer budget;
that is the user's decision.

## Budget and placement

- Host: loulou RTX 5090 (idle 2026-11-14; trex and fifi also idle, riri busy).
- Readiness: prepare plus two HWA updates with full 256-sequence selection; ≤ 0.5 GPU-h.
- Main run: one five-pass std3 arm; cap 45 GPU-h including retries.
- Storage: ≤ 20 GiB (final and best weights about 1.7 GB each, training state about 7 GB).
- Remote root: `/home/filip/analog-research/opt-cmo-drn-l12-v1/` on loulou.
- The shared run-watch queue owns launch, monitoring and validation. The run supports exact resume.

## Handoff

- Launch commit `9e1c024` (merged into `refactor/training-core` and pushed).
- Local pinned checkout: `results/opt125m-cmo-drn-l12-hwa-v1/source-9e1c024`.
- Remote root on loulou: `/home/filip/analog-research/opt-cmo-drn-l12-v1/`.
  - `source-9e1c024/` is a read-only `git archive` export with `source_snapshot.json`
    (`bfcca578…`, from `campaigns/drift-regimes/launch/make_receipt.py`).
    `ebl.lifecycle.staging` cannot stage this run: its root resolves to `ebl/`
    (on-chip-learning #41).
  - `inputs/ed53b6cb….pt`: the 12-MLP source, read-only, hash checked.
  - The teacher and corpus are the existing `opt-cmo-exp1/inputs/` files, hashes checked.
- Queue job specs: `results/opt125m-cmo-drn-l12-hwa-v1/jobs/`. The shared run-watch
  queue owns execution and validation.
- Inspect a job: `python3 ~/.codex/skills/run-watch/scripts/gpu_queue.py status <id>`.

| Job | State | Notes |
| --- | --- | --- |
| `opt-cmo-drn-l12-prepare-v1` | complete, validated | 10.6 s. Mapped `f1961b3f…` (1.18 GB). Clean development KL 0.1266 over 256 sequences (PPL 8.19). All denominators ≥ 240. Max voltage 12.9 in layer 0 against ≤ 8.4 in the other blocks (6-MLP: ≤ 5.1). |
| `opt-cmo-drn-l12-hwa-readiness-v1` | cancelled before launch | malformed `validate_argv` in the spec; never ran |
| `opt-cmo-drn-l12-hwa-readiness-v2` | running | two std3 updates with full selection |
| `opt-cmo-drn-l12-hwa-std3-v1` | prepared | submit after readiness passes; budget from the measured throughput |
