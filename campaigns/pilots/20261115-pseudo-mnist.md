# Ideal pseudoconductance attention on MNIST

User requested implementation, training like yesterday, capped iterations and
EP/BPTT agreement. Clarification pending: yesterday's recorded task was masked
completion, whereas the new message says permuted MNIST. The attention block
and gradient verification are independent of that choice.

Model: separate anchored stars for every query/head, positive input encoding
x=1+v, h=1, h0=.01, supplied controls a from digital QK/softmax. Values bounded
with .75*tanh. No sharing of free key states between queries. Ideal encoded
KCL, not BSIM3 or a physical capacitance/time simulation. Block Gauss-Seidel
relaxation updates all key states then output state. Fixed caps 4/8/16/32/64/128
for diagnostics; primary training cap 32. Each EP backward uses two phases
of 32 sweeps starting at the same free state and a frozen output-error current.
Digital BP handles the scorer, encoder/decoder and residual/FF layers.

Comparison: identical-initialization EP and BPTT training, 2000 updates each,
batch64, Adam .003, seed20261114, 16 tokens/D64/four heads, matching yesterday's
reference shape and 55k/5k/10k split. Initialization/final fixed-cohort gradient
comparison globally and by parameter group; exact closed-form equilibrium is
an independent reference, never substituted for the training forward.
Report cosine and relative L2, forward error and KCL residual, plus learning
curves and full validation/test coverage. Target cosine>=.99 and relative L2
<=.05 at cap32; failures are evidence, not silently repaired by more sweeps.
Task metrics improve on initial and a task-specific baseline; digital-parity
or advantage is measured, not assumed. No accuracy threshold guarantees success.

Local RTX3090, verified idle/free23GB, FP32 training, FP64 algebra tests.
Budget <=60 GPU minutes total including readiness/retries, <=500MB raw results
under /tmp/pseudo-mnist-results (home has ~116MB free); compact evidence copied
to backend results at collection. No new remote/GPU host, noise or tuning sweep.
Current agent owns short-run monitoring, metrics every50 updates, five-minute
stall gate, and retains all failed attempts. Durable watcher needed if handing
off a long run. Final records in this pilot and shared exploration.

## Task resolution and readiness

No answer to the task clarification arrived during independent implementation.
Cover both interpretations explicitly within the original 60-minute budget:
completion-ep, completion-bptt, permuted_classification-ep and
permuted_classification-bptt, each2000updates. Classification uses one fixed
784-pixel permutation(seed3105) across all splits, then16 patches, token-mean
readout and cross entropy. Completion exactly reuses yesterday's split/mask
helpers (copied to a separately owned module from energy-transformer source),
half-masked image MSE, original unpermuted pixel layout and mean-image baseline.
Classification's baseline is always class0 (~10%). Four independently named
native bundles; no conflation of classification accuracy and completion MSE.
FP32, no TF32, 32 full key/output block sweeps in every free/nudged phase.

Controls are digital softmax(QK), positive with row sum1; this has no per-edge
sharpness cap. Couplings are a, effective weights a/(1+a). This fixes the
total input conductance and gives the block-GS contraction at most1/(2*1.01),
so small caps should suffice even for sharp attention. Neither that solver nor
this normalization is evidence of physical circuit latency. All other digital
maps use BP. EP is a centered local energy-statistic readout with frozen incoming
error: current amplitude is scaled to max abs .005 and the scaling is undone
in the gradient. This avoids sub-ulp FP32 nudges and is valid for this linear
conditional circuit; it is not nonlinear full-network loss-nudged EP.

Two FP64 tests passed: settling/exact/EP/BPTT/FD agreement and a deliberately
short finite-time counterexample with proper summation over duplicated value
inputs. CUDA startup first failed exactly as recorded yesterday; retry passed.
Supervisor retains a tiny CUDA tensor/context across child runs, using the
already established local workaround. Initial GPU32-sweep global EP/BPTT
cosine0.9999999999994, relative L2 1.12e-6. Full public-CLI two-update smokes
for each task are under /tmp/pseudo-mnist-readiness. Failed readiness is retained.

Commands, one task-method config per bundle:

```bash
python -m ebl train --config examples/pseudo_mnist/completion-ep.json --output-dir /tmp/pseudo-mnist-results/completion-ep
python -m ebl train --config examples/pseudo_mnist/completion-bptt.json --output-dir /tmp/pseudo-mnist-results/completion-bptt
python -m ebl train --config examples/pseudo_mnist/permuted_classification-ep.json --output-dir /tmp/pseudo-mnist-results/permuted_classification-ep
python -m ebl train --config examples/pseudo_mnist/permuted_classification-bptt.json --output-dir /tmp/pseudo-mnist-results/permuted_classification-bptt
```

Do not start a further900-second case with less than900seconds of the shared
60-minute allowance remaining; readiness and retries consume it. An early
CUDA-initialization failure may retry once with identical config and preserved
log/bundle; no other silent retry. Source is pinned after merge/push. Agent
monitors the serial supervisor, checks growing metrics, and reviews all results.
