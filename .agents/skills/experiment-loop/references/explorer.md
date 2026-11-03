# Explorer: from a question to a frozen comparison

Output: the pilot note's "Question and decision", "Frozen comparison" and "Budget
and execution" caps, in `campaigns/pilots/<yyyymmdd>-<slug>.md` built from
[the pilot template](pilot.md) (or a campaign experiment note for sustained work).
State `planned`, or `ready` once every Explorer field is resolved. The Explorer
writes no campaign lifecycle and launches nothing. Choices come only from the
[lifecycle contract](../../../../docs/crossbar_lifecycle.md); this file is the
procedure, not the menu.

## 1. Prior evidence

Search before designing: `campaigns/*/series/*/results/`, `campaigns/pilots/*.md`
and `docs/*_results.md` (search the network, technology, defect, noise, drift and arm names).
Campaign result notes carry a one-line `summary` in their frontmatter; read those first.
Write a short linked list of what is already known and under which regime. If the
question is already answered in that regime, stop and say so; if a related result
exists, state what differs. For any question that touches HWA training, also read
the two reference papers in [`hwa_papers/`](../../../../hwa_papers/README.md) (Rasch
et al. 2023; Büchel et al. 2025, *Analog Foundation Models*) and note what they
recommend for the choice at hand.

## 2. Question and decision

- One question comparing interventions under declared corrupted-device,
  write-noise or drift conditions, with the outcome and physical cost explicit.
- The decision it informs (continue, scale, change regime, stop).
- The expected outcome and at least one alternative that would change the decision.

## 3. Frozen comparison

On the experiment's `exp/<id>` branch, walk every Explorer field of the ownership
table in [SKILL.md](../SKILL.md). Mark each choice `proposed` or `confirmed by user`.
The user confirms material choices: network/analog subset, corpus, metric,
defect/noise/drift regime, comparison arms and budget. Record the selected values
in the note; the Codifier transcribes them into the lifecycle.

- State which effects are enabled, absent or varied: corrupted devices, write
  noise and drift. Declare their interactions when combined.
- Specify technology, model revision, topology/mapping and conductance bounds.
  For defects, record kinds, rates and placement. For programming and subsequent
  writes, record the noise law and strength, affected devices, verify policy and
  clipping. For drift, record the law, age grid, temporal correlations and
  post-write reset behavior. Label idealized controls and extrapolations.
- Select the network, analog subset, teacher and initialization with pinned inputs.
  Declare corpus, sequence length, disjoint adaptation/evaluation cohorts,
  objective, primary metric and selection rule.
- Declare HWA sources and settings, calibration, rewrite and learning controls,
  update laws and physical cost measurements. For HWA choices consult
  [`hwa_papers/`](../../../../hwa_papers/README.md), recording which follow the
  papers and which deviate, with reasons.
- Declare independent assignment and selection arrays, programming realizations
  and trajectory pairing. A one-assignment pilot cannot establish generality.
- Freeze learning rates and epochs from cited evidence or a separate development
  lifecycle, without tuning on the evaluation split.

Select a compatible template from `examples/lifecycles/` after these choices are
resolved. Template values and prior campaign defaults do not choose the next run's
science. Link and pin any shared protocol the experiment adopts.

## 4. Decision rule

Fill and freeze before any result exists; thresholds are labelled `proposed` until
the user confirms them. Adapt this teacher-KL example to the declared metric,
time aggregation and physical cost units.

```text
Unit: one assignment x one declared defect/noise/drift condition.
Keep distinct conditions separate.
gap closed = (initial - final) / (initial - clean), in teacher KL on <split>
Matched effect: learn arm vs the hold arm with the same calibration flag and HWA source.
Viability: best learn row vs the best no-weight-learning row (none, calibration,
  rewrite) across the declared HWA sources, same assignment and condition.
On-chip is viable for <regime> if, in at least <k> of <n> assignments,
  gap closed improves by >= <delta> over that best alternative at <= <cost>
  pulses per cell. Inconclusive if <...>. Report cost for every learn row.
```

## 5. Confound check

- Teacher, analog subset, corpus, cohorts, minibatch order, objective, seeds and
  budget are identical across arms; only the declared intervention differs.
- Programming realizations of one assignment are not independent arrays.
- Temporal pairing and post-write randomness match the declared model and time grid;
  a shared seed alone does not establish matched trajectories.
- HWA selection uses selection arrays and development data, never test data.
- A smoke or probe is not evidence; partial coverage stays labelled partial.
- Any change of corpus, tokenization or cohort between arms invalidates the comparison.

## 6. Budget caps

Copy the template to a scratch path, set the proposed fields and `max_examples` 0,
and run `python -m ebl lifecycle describe <draft> --json` (it launches nothing). Record stage runs,
trajectories, analog weights and the state estimate. Then set caps: compute hours,
storage GB, deadline, stopping conditions and checkpoint retention. Wall-clock time
stays unknown until the Codifier's cost probe; say so in the note.

## 7. Supportability

Check every choice against the contract's "Not yet supported" list. Redesign within
the supported menu, or record `needs extension: <point>` from its "Extension points"
table; an extension is Codifier work, not a blocker.

## 8. Hand off

Hand the note to the Codifier only when every Explorer field is resolved and the
material ones are confirmed. Escalate scientific choices to the user instead of
choosing silently. The note lives on the experiment's own branch from its first
draft; commit it there before handing off ([git checkpoints](../SKILL.md#git-checkpoints)).
