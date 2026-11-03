# Prepare exact inputs and checks

Follow the skill's [Codifier procedure](../.agents/skills/experiment-loop/references/codifier.md)
and [git checkpoints](../.agents/skills/experiment-loop/SKILL.md#git-checkpoints).
Build the strict lifecycle in the assigned branch. Pin teacher, corpus, source,
initial state and native bundles by explicit identities.

```bash
python -m ebl lifecycle check
python -m ebl lifecycle describe campaigns/<campaign>/lifecycles/<id>.json --json
python -m ebl lifecycle plan campaigns/<campaign>/lifecycles/<id>.json \
  --teacher-weights <pinned-teacher> --output <plan-root> --device cpu
```

Check the actual [runtime contract](../docs/experiment_runtime.md), supported
resume capability, apparent-state forwards and declared write costs. Run bounded
local checks proportionate to the change; measure representative state/memory
before scaling. A plan or dry run is not numerical validation or launch authority.
Record the exact target, commands, paths, expected progress and budget in the handoff.
