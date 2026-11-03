# exp-021 and exp-022 lifecycles (archived)

These 29 lifecycles come from branch `codex/exp021-onchip-lr-screen`, which was never
merged into `codex/revised-hwa-training`. They were copied here on 3 November 2026,
before the branch and its worktree were removed. The code that ran them is kept at the
tag `archive/exp021-onchip-lr-screen`. The branch also added two exp-016 lifecycles
(`opt125m-cmo-l6-exp16-pair-rewrite-std3-kinds-{learn,sweep}`). Identical copies
were already in `../../lifecycles/`, where the exp-016 note expects them, so they are
not repeated here.

The `exp21-` prefix is also used by `../../lifecycles/opt125m-cmo-l6-exp21-fault-placement*`
and `opt125m-cmo-l6-exp21-array-rewrite`. Those belong to the
[fault-placement pilot](../../../pilots/20261016-fault-placement.md), not to exp-021.

They sit outside `lifecycles/` because 14 of them declare `onchip.lr_schedule`. That
option exists only at the archive tag, so this tree's parser rejects them. Plan or rerun
them from a checkout of the tag, not from this tree.

- Experiment notes: [exp-021](../../series/002-onchip-recovery/experiments/exp-021-onchip-lr-screen.md),
  [exp-022](../../series/002-onchip-recovery/experiments/exp-022-pv-learning-noise.md)
- Result notes: [exp-021](../../series/002-onchip-recovery/results/exp-021.md),
  [exp-022](../../series/002-onchip-recovery/results/exp-022.md)
- Raw evidence (ignored): `results/lifecycles/opt125m-cmo-l6-exp21-*` and
  `results/lifecycles/opt125m-cmo-l6-exp22-pv-learning/`
