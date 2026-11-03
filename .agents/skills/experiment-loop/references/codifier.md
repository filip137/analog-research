# Codifier: from a frozen comparison to a running lifecycle

Input: a note whose Explorer fields are resolved. Output: the lifecycle file, any
missing capability with its test, cost-probe measurements, the plan, a launch only
under assigned execution, and the note's handoff. Read the
[lifecycle contract](../../../../docs/crossbar_lifecycle.md) fully and the
[execution policy](../../../../docs/experiment_workflow.md) for launch rules. Never
change a scientific setting to make something fit or pass.

## 1. Spec check

Work on the assigned experiment branch. Compare its note with the ownership
table in [SKILL.md](../SKILL.md), including enabled defects, write noise and drift.
Every Explorer field must be resolved and the material ones confirmed by the user. If not,
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
- OM populations: `EBL_AIHWKIT_PYTHON` must name the AIHWKit 1.1.0 interpreter
  (OM only; CMO and PCM are analytic).

## 4. Lifecycle file

Use the template in `examples/lifecycles/` that matches the note's declared
regime, and save the exact definition on the experiment branch at
`campaigns/<campaign>/lifecycles/<lifecycle_id>.json` (file name equals the id).

- Explorer fields: transcribe verbatim from the note, including each selected
  device effect and its settings. A template never supplies an unresolved choice.
- Codifier fields: choose explicit seeds with disjoint roles and record them.
  Preserve the note's physical array size when placement depends on it;
  otherwise choose computational tiling within the engineering allowance.
  Use `max_examples` 0 for evidence.
- Run `python -m workflow check` and `python -m workflow describe <file>`; paste the
  describe summary (stages, trajectories, analog weights, state estimate) into the
  note. If the estimate exceeds the Explorer's storage cap, escalate before going on.

## 5. Cost probe

Required the first time a technology, number of analog layers or batch size runs.
Write `<lifecycle_id>-probe.json`, labelled as a probe in its question, retaining
the declared network, device effects and execution stages. Reduce assignments,
sources, cohorts and updates within the probe allowance, and cover representative
control/impaired conditions from the assigned comparison. A drift-only probe does
not acquire defects or HWA training simply because a template includes them.
Saved state per trajectory does not shrink with `max_examples`, so check the
probe's own `describe` estimate before running.

Plan and run it locally (steps 6 and 7 with the probe). From each stage's
`result.json` read `elapsed_seconds` and `peak_gpu_memory_bytes`; for CPU runs, time
the costliest stage once with `/usr/bin/time -v python -m ebl train ...` for maximum
memory. Measure state sizes with `du` on `checkpoints/`. Extrapolate each stage by
its examples x epochs (plus evaluation and P&V per case and array) and the real
number of sources, cases and arrays, and compare with the caps. Over a cap: escalate
to the Explorer/user; do not shrink cohorts, arrays, cases or epochs on your own.

## 6. Plan and dry run

```bash
RUN=/home/filip/analog-research/hwa/results/lifecycles/<id>   # absolute, never inside a worktree
python -m workflow plan campaigns/<campaign>/lifecycles/<id>.json \
  --teacher-weights <teacher> --output $RUN/plan \
  [--device cuda:N] [--cpu-threads K]
python -m ebl campaign run --manifest $RUN/plan/campaign.json \
  --output-dir $RUN/runs --dry-run
```

Run these in the checkout that will execute, because the plan pins it: the
experiment worktree for a probe or smoke, the pinned launch checkout of step 7 for
evidence. Add `--allow-dirty` only for probes or smokes from an uncommitted tree,
and say so.

## 7. Launch gate

Before launching, show the user the cases, target host/device, budget, expected
duration and output paths. Then take the launch [git checkpoint](../SKILL.md#git-checkpoints):
commit code, tests, lifecycles and the note on `exp/<id>`, integrate into the trunk
and push, create the pinned checkout of that commit, and plan from it (step 6)
without `--allow-dirty`. Remote hosts check out the same pushed commit. Launch only
under assigned execution, in a persistent session (e.g. tmux), with
`python -m ebl campaign run ... --resume`, and record the session, PID and log path.
Verify initial semantic progress, not just a live process:

- `prepare`: complete, with digital `teacher_kl` at most 1e-6 (the gate fails otherwise);
- `devices`: `kernel_adequate` true (OM);
- each training `hwa__<arm>` (method `hwa`): an `hwa_selection` record in `metrics.jsonl`.

## 8. Handoff and failures

Fill the note's "Current execution and monitoring handoff", then commit it on
`exp/<id>`, again at each material change: branch, launch commit and pinned
checkout, cases, lifecycle and
plan paths with digests, run root, session/PID/log, expected stage order and
durations from the probe, next check, deadline and remaining budget, inspect and
collect commands (`python -m ebl runs inspect RUN_DIR --verify-artifacts
--require-complete`; `python -m workflow collect <runs> --output <collected>`) and the
escalation route. Retry only understood operational failures, within the same budget,
keeping every failed attempt; retries are not replicates.
