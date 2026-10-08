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
and `docs/*_results.md` (grep the network, technology, defect kind and arm names).
Campaign result notes carry a one-line `summary` in their frontmatter; read those first.
Write a short linked list of what is already known and under which regime. If the
question is already answered in that regime, stop and say so; if a related result
exists, state what differs.

## 2. Question and decision

- One question, phrased as a comparison, e.g. "With N analog MLPs and gmax
  fractions F, does on-chip learning close more of the teacher-KL gap than
  calibration or rewrite, and at what pulse cost?"
- The decision it informs (continue, scale, change regime, stop).
- The expected outcome and at least one alternative that would change the decision.

## 3. Frozen comparison

Walk every Explorer field of the ownership table in [SKILL.md](../SKILL.md). Mark
each choice `proposed` or `confirmed by user`; the user confirms the material ones:
corpus, metric, number of analog layers, defect regime, comparison arms and budget.
Starting proposals for OPT, from `examples/lifecycles/opt125m-om-mlp4-template.json`:

- network: `opt_mlp_suffix`, `analog_decoder_layers` N (1–2 for a first probe);
- technology `om`, closed-loop P&V as in the template;
- defects: a `none` case plus the chosen fractions of one kind per question,
  e.g. `gmax` at 10000, 20000, 50000 ppm; one list, never pooled;
- HWA sources: `digital` and `standard_hwa`; add `cdt_<kind>` only when
  corruption-aware training is part of the question;
- on-chip arms: the five standard ones (`none`, `calibration`, `rewrite`,
  `onchip_weights`, `onchip_calibration`), `om_closed_loop_pulse_adam`;
- arrays: one assignment for a labelled pilot, at least three for any claim;
  selection arrays separate from assignments;
- metric: `teacher_kl` primary, perplexity reported; corpus, sequence length and
  cohort sizes declared, with adaptation and evaluation tokens disjoint;
- learning rates and epochs: cited from an earlier frozen result or chosen by a
  separate development lifecycle; never tuned on the evaluation split.

## 4. Decision rule

Fill and freeze before any result exists; thresholds are labelled `proposed` until
the user confirms them.

```text
Unit: one assignment x one defect case. Never pool across defect rates.
gap closed = (initial - final) / (initial - clean), in teacher KL on <split>
Matched effect: learn arm vs the hold arm with the same calibration flag and HWA source.
Viability: best learn row vs the best no-weight-learning row (none, calibration,
  rewrite) across the declared HWA sources, same assignment and case.
On-chip is viable for <regime> if, in at least <k> of <n> assignments,
  gap closed improves by >= <delta> over that best alternative at <= <cost>
  pulses per cell. Inconclusive if <...>. Report cost for every learn row.
```

## 5. Confound check

- Teacher, analog subset, corpus, cohorts, minibatch order, objective, seeds and
  budget are identical across arms; only the declared intervention differs.
- Programming realizations of one assignment are not independent arrays.
- HWA selection uses selection arrays and development data, never test data.
- A smoke or probe is not evidence; partial coverage stays labelled partial.
- Any change of corpus, tokenization or cohort between arms invalidates the comparison.

## 6. Budget caps

Copy the template to a scratch path, set the proposed fields and `max_examples` 0,
and run `python -m workflow describe <draft> --json` (it launches nothing). Record stage runs,
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
choosing silently.
