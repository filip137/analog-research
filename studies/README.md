> Workflow migration (2026-09-25): the [question-led policy](../docs/experiment_workflow.md)
> supersedes the legacy planning/global-manifest requirements below. Existing study
> formats remain supported; new pilots use one note and native run bundles.

# Study plans

This directory contains small, tracked scientific plans for new exploratory
studies. Raw runs stay below ignored `results/<study-id>/`; the plan remains in
Git and freezes the initial hypothesis, declared configs, completion criteria,
and analysis plan before execution.

Start with `example-exploratory.json`, give a real study a stable kebab-case
ID, and prepare it with:

```bash
python -m ebl study prepare --plan studies/<study-id>.json
```

One config corresponds to one expected completed run. Put separate seeds or
scientific settings in separate config files. The same config may be retried
after an operational failure, but two successful runs for one declared config
make coverage ambiguous and block finalization.

The review template belongs in `results/<study-id>/analysis/review.json` only
after the declared evidence has been collected and inspected.

EP/DRN/LoRA and MNIST crossbar v1 execution plans are available at `16938bf`.
Their frozen results and scientific notes remain historical evidence. This directory
keeps current crossbar/teacher/characterization plans and reference receipts;
see [supported scope](../docs/crossbar_scope.md).
