# Token-selective drift recovery — issue #25

User-authorized 11 November 2026: crossbar LoRA, Jean Zay V100 dev queue.
Scientific contract and current handoff:
https://github.com/filip137/on-chip-learning/blob/main/campaigns/recovery-data-preparation/series/002-token-backward/experiments/exp-003-token-backward.md

Isolated branch `exp/recovery-token-backward` from the validated target-count
commit `ad902b0`. This extends the existing native drift-regimes instrument;
it does not retrofit the maintained trunk's architecture migration.

Generate strict candidates with `campaigns/drift-regimes/make_token_backward_lifecycles.py`.
Use the existing `launch/worker.py` for plan, run, inspect and collection.
`launch/make_token_manifest.py` maps immutable candidates to groups;
`launch/select_token_rates.py` selects development-only rates and enforces
the sample-8 historical reference. Cap: 8 V100-hours, 100 GB, <= 2 h/job.

The targeted CPU checks cover forward equivalence, the stopped-gradient oracle,
all-position gradients, causality, saved-tensor storage, accumulated updates,
native lifecycle execution and unchanged historical target paths.
