# Codifier: from a frozen comparison to a running lifecycle

Input: a note whose Explorer fields are resolved. Output: the lifecycle file, any
missing capability with its test, cost-probe measurements, the plan, a launch only
under assigned execution, and the note's handoff. Read the
[lifecycle contract](../../../../docs/crossbar_lifecycle.md) fully and the
[execution policy](../../../../docs/experiment_workflow.md) for launch rules. Never
change a scientific setting to make something fit or pass.

## 1. Spec check

Compare the note with the ownership table in [SKILL.md](../SKILL.md). Every
Explorer field must be resolved and the material ones confirmed by the user. If not,
return the list of open fields to the Explorer; do not fill them.

## 2. Supported check

Map each choice onto a lifecycle field. If one is not supported, it is one of the
contract's "Extension points": change the named module, add a contract or
equivalence test, run `python -m pytest tests/test_workflow_*.py`, and record "What
changed" in the note. Never add a per-study runtime or bypass the stages.

## 3. Inputs

- Teacher: the pinned checkpoint, e.g. for OPT
  `~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/<revision>/pytorch_model.bin`
  with the revision and SHA-256 in `workflow/networks/opt_mlp.py`.
- Corpus (OPT): `$EBL_CORPUS_ROOT/<name>.pt` with the declared SHA-256
  (`sha256sum`). If absent, build it from the note's text files with
  `python -m workflow corpus --name <name> --model-dir <snapshot> --train ... --validation ... --test ...`
  and copy the printed digest into the note and the lifecycle.
- OM populations: `EBL_AIHWKIT_PYTHON` must name the AIHWKit 1.1.0 interpreter.

## 4. Lifecycle file

Copy `examples/lifecycles/opt125m-om-mlp4-template.json` to
`campaigns/<campaign>/lifecycles/<lifecycle_id>.json` (file name equals the id).

- Explorer fields: transcribe verbatim from the note.
- Codifier fields: assignment seeds 271001, 271002, ...; selection seeds 211001, ...;
  `endpoint_seed_offset` 10000; characterization seeds 231001/231002 (the parser
  rejects collisions); `tile_size` 512; `max_examples` 0 for evidence.
- Run `python -m workflow check` and `python -m workflow describe <file>`; paste the
  describe summary (stages, trajectories, analog weights, state estimate) into the
  note. If the estimate exceeds the Explorer's storage cap, escalate before going on.

## 5. Cost probe

Required the first time a technology, number of analog layers or batch size runs.
Write `<lifecycle_id>-probe.json`, labelled as a probe in its question: the same
network, devices and on-chip arms, one assignment, one HWA source that trains, the
`none` case plus one corrupted case, `max_examples` 8 and one epoch per stage.
Saved state per trajectory does not shrink with `max_examples`, so check the probe's
own `describe` estimate first; the reduced sources and cases keep it near 4 GB per
analog decoder layer (OM).

Plan and run it locally (steps 6 and 7 with the probe). From each stage's
`result.json` read `elapsed_seconds` and `peak_gpu_memory_bytes`; for CPU runs, time
the costliest stage once with `/usr/bin/time -v python -m ebl train ...` for maximum
memory. Measure state sizes with `du` on `checkpoints/`. Extrapolate each stage by
its examples x epochs (plus evaluation and P&V per case and array) and the real
number of sources, cases and arrays, and compare with the caps. Over a cap: escalate
to the Explorer/user; do not shrink cohorts, arrays, cases or epochs on your own.

## 6. Plan and dry run

```bash
python -m workflow plan campaigns/<campaign>/lifecycles/<id>.json \
  --teacher-weights <teacher> --output results/lifecycles/<id>/plan \
  [--device cuda:N] [--cpu-threads K]
python -m ebl campaign run --manifest results/lifecycles/<id>/plan/campaign.json \
  --output-dir results/lifecycles/<id>/runs --dry-run
```

Add `--allow-dirty` only for probes or smokes from an uncommitted tree, and say so.

## 7. Launch gate

Before launching, show the user the cases, target host/device, budget, expected
duration and output paths. Launch only under assigned execution, in a persistent
session (e.g. tmux), with `python -m ebl campaign run ... --resume`, and record the
session, PID and log path. Verify initial semantic progress, not just a live process:

- `prepare`: complete, with digital `teacher_kl` at most 1e-6 (the gate fails otherwise);
- `devices`: `kernel_adequate` true (OM);
- each training `hwa__<arm>` (method `hwa`): an `hwa_selection` record in `metrics.jsonl`.

## 8. Handoff and failures

Fill the note's "Current execution and monitoring handoff": cases, lifecycle and
plan paths with digests, run root, session/PID/log, expected stage order and
durations from the probe, next check, deadline and remaining budget, inspect and
collect commands (`python -m ebl runs inspect RUN_DIR --verify-artifacts
--require-complete`; `python -m workflow collect <runs> --output <collected>`) and the
escalation route. Retry only understood operational failures, within the same budget,
keeping every failed attempt; retries are not replicates.
