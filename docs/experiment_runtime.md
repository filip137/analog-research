# Crossbar experiment runtime

The public entry point is `python -m ebl`. It resolves a strict JSON config through
`experiments/definitions.py` and dispatches to the selected runtime. Unknown
settings and unsupported modes fail before the numerical runtime loads.
[Supported families](crossbar_scope.md) include crossbars, digital teacher controls
and device characterization.

```bash
python -m ebl describe --json
python -m ebl describe --experiment mnist_ibm_om_crossbar_relu.v2 --json
python -m ebl train --config CONFIG.json --output-dir RESULTS --teacher-weights TEACHER.pt
python -m ebl validate --config examples/mnist_relu/teacher.json --output-dir RESULTS --weights TEACHER.pt
python -m ebl characterize --config CONFIG.json --output-dir RESULTS
python -m ebl runs inspect RUN_DIR --verify-artifacts --require-complete --json
```

Input requirements depend on experiment and stage. `--weights` and `--resume`
are mutually exclusive. Crossbar families can require explicit teacher weights,
device data/cached activations, models/populations, deployed state and frozen
selection receipts. The runtime enforces stage-specific requirements; CLI options
do not override scientific settings. See staged MNIST configs and the
[CIFAR reproduction guide](cifar_reproduction.md) for exact commands.

Each numerical invocation creates a native bundle containing resolved config,
source/runtime identity, status, metrics, artifact hashes and result metadata.
Checkpoints preserve named keys and restore tensors in place. Existing named-weight,
resume and population schemas remain compatible.

`campaign run --manifest ... --output-dir ...` orchestrates exact configs and
explicit inputs through subprocesses, with dry runs and completed-stage reuse.
Campaign lifecycles (`crossbar_lifecycle.v1`) are planned into such a manifest by
`python -m workflow plan`; see the [lifecycle contract](crossbar_lifecycle.md).
Optional `study prepare`, `study summarize` and `study finalize` commands retain
study inspection and local review receipts. Current work follows the
[experiment workflow](experiment_workflow.md); cleanup does not update global
historical ledgers.

Retired EP/DRN/LoRA experiments and the older MNIST crossbar v1 runtime remain at
commit `16938bf`. This branch no longer exposes `linspace`, `checkpoint import-legacy`
or `--base-weights`. Historical native inspection is read-only and does not require
the old experiment to remain registered.
