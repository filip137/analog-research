# Architecture migration checks

These are bounded engineering workloads, not scientific campaigns. The
[migration pilot](../campaigns/pilots/20261103-architecture-boundaries.md) records
source revisions, exact comparisons, measurements and limitations.

`native_boundaries.py` constructs synthetic OPT networks and native CMO/drift
state. `tiny` has 2,048 analog weights; `opt_mlp` has one OPT-125M-width MLP with
4,718,592 weights, a reduced vocabulary and short synthetic sequences. It is not
a complete OPT-125M training workload. All work is CPU FP32, one thread, fixed
seeds, three warm-ups and ten timed repetitions. State restoration is outside
the operation timings. The native trace includes logits, gradients, device and
optimizer state, drift, RNG and cost.

From each pinned checkout, use the same interpreter and environment:

```bash
CUDA_VISIBLE_DEVICES='' MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 <python> -B -m benchmarks.native_boundaries \
  --workload opt_mlp --output <evidence>/<version>-<process>
```

Repeat in three fresh processes per version/workload, alternating version order.
Do not run tests concurrently with timings. Add `--profile` only to a separate
profiling invocation. Report the median and range of process medians, within-process
IQR, persistent storage, checkpoint bytes, and process peak RSS. RSS includes
imports and allocator behavior; it is not GPU peak memory.

For exact OPT traversal and hierarchical fault comparisons, use
`capture_opt.py` / `capture_faults.py` with explicit `baseline` or `refactor`
source layout and an output path. The capture harness can be supplied on stdin
to a pinned older checkout which does not contain it:

```bash
cd <pinned-baseline>
<python> -B - baseline <evidence>/opt-before.pt < <current-source>/benchmarks/capture_opt.py
cd <current-source>
<python> -B -m benchmarks.capture_opt refactor <evidence>/opt-after.pt
<python> -B -m benchmarks.compare_exact <evidence>/opt-before.pt <evidence>/opt-after.pt
```

Repeat for `capture_faults.py`. These explicit revision adapters belong only to
the engineering harness; production imports have no legacy fallback. No tolerance
is used for migration replay. `verify_shared.py <hwa> <drn>` checks the declared
shared source bytes without importing either backend's runtime.
