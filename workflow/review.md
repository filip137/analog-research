# Collect and review

```bash
python -m ebl runs inspect <run-dir> --verify-artifacts --require-complete
python -m ebl lifecycle collect <runs-root> --output <collected-root>
```

Inspect explicit bundles, reconcile assigned cases, and distinguish process
completion, integrity, scientific validity, coverage and review. The collector
flags duplicate coverage; retries are not independent replicates. Do not pool
different defect, drift or device regimes.

Apply the note's frozen decision rule. Report the analog identities, held-state
semantics, random coupling and physical write/read accounting alongside outcomes.
Separate observed measurements from inference and unresolved choices. Preserve
negative and partial evidence. Record interpretation in the pilot/result note,
then complete the skill's [closing git checkpoint](../.agents/skills/experiment-loop/SKILL.md#git-checkpoints).
Historical global manifests and personal notes are not closeout destinations.
