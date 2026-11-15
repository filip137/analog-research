# Ideal pseudoconductance attention on MNIST

User requested implementation, training like yesterday, capped iterations and
EP/BPTT agreement. Yesterday's recorded task was masked completion, whereas the
new message says permuted MNIST. Both interpretations were run separately within
the declared allowance; the implementation and full review below are complete.

Model: separate anchored stars for every query/head, positive input encoding
x=1+v, h=1, h0=.01, supplied controls a from digital QK/softmax. Values bounded
with .75*tanh. No sharing of free key states between queries. Ideal encoded
KCL, not BSIM3 or a physical capacitance/time simulation. Block Gauss-Seidel
relaxation updates all key states then output state. Fixed caps 4/8/16/32/64/128
for diagnostics; primary training cap 32. Each EP backward uses two phases
of 32 sweeps starting at the same free state and a frozen output-error current.
Digital BP handles the scorer, encoder/decoder and residual/FF layers.

Comparison: identical-initialization EP and BPTT training, 2000 updates each,
batch 64, Adam .003, seed 20261114, 16 tokens/D64/four heads, matching yesterday's
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
The executing agent owned monitoring through completion, with metrics every50
updates and a five-minute stall gate. No jobs remain active. Final records in
this pilot and the shared exploration; worktrees close after published review.

## Task resolution and readiness

No answer to the task clarification arrived during independent implementation.
Cover both interpretations explicitly within the original 60-minute budget:
completion-ep, completion-bptt, permuted_classification-ep and
permuted_classification-bptt, each 2,000updates. Classification uses one fixed
784-pixel permutation(seed 3105) across all splits, then16 patches, token-mean
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

## Final review

All four 2,000-update runs completed without retry, with exactly 32 sweeps per
free/positive/negative phase. Launch5d6e18b152b6b8c5b260c3c2cf9a19e653bc112e
was clean and published before execution. Pinned source/tmp/pseudo-mnist-source;
supervisorPID3540830, exec session41134, now exited successfully. Supervisor
wall 98.24 s, combined native run time 92.44 s, plus 300 s conservatively charged
readiness. Storage and60-minute allowance respected; no remote execution.

| Full 10k test | EP | BPTT | Baseline |
| --- | ---: | ---: | ---: |
| Permuted classification accuracy | 90.06% | 89.40% | 9.80%, always0 |
| Completion MSE | .044297515 | .044639052 | .067346309, training mean |

Full 5k validation: classification89.74%/88.58%, completion.04523489/.04555981.
Every paired initialization tensor and permutation matched exactly; sampling,
masks, optimizer and data splits are matched. This one seed supports learning
with the EP block and comparable task quality, not EP superiority. Completion
has35,008parameters and classifier32,512; each has16tokens,width 64,4heads.

At 32 sweeps, all initial/final gradient checks pass globally and by parameter
group on three fixed8-example validation minibatches per checkpoint. Worst
final cosine across EP/BPTT,EP/exact,BPTT/exact is.999999954; worst relative L2
is.000304665 (0.03047%). Per-arm worst relative errors are.03047%,.02830%,
.01374%,.01733% for completion-EP,completion-BPTT,classifier-EP,classifier-BPTT.
Largest final KCL residual2.384e-7. The exact operator is diagnostic only;
training forwards use the actual capped iterative state.

The cap sweep catches a material distinction:8 sweeps has EP/BPTT error<=.401%
but up to10.94% error against exact equilibrium (completion-BPTT checkpoint,
batch0,ff1.weight,EP/exact). Four sweeps reaches6.51% EP/BPTT and21.51% against
exact. Sixteen is the smallest tested cap passing all three comparisons:
worst error.03041%, minimum cosine.999999954. Therefore retain 32as the validated
training setting;16 is a sampled diagnostic candidate, not a trained result.
Tiny finite-forward changes can affect downstream gradients; agreement of
two finite-time methods alone is insufficient. No claim covers every untested
batch or intermediate checkpoint.

Attention ablations on the same256-image probe: EP completion normal.0451701,
uniform.0735435,zero.1000596; EP classifier normal90.234%,uniform22.656%,zero
23.438%. These inference interventions support use of learned routing; they
are not separately trained controls. Train/validation/test are55k/5k/10k;
test counts and native hashes verified. Two numerical regression tests and
both public-CLI task smokes pass. Initial CUDA failure occurred before bundle
creation; the transcript/record preserves the incident, no failed training
bundle is claimed. Subsequent supervision used the prior context workaround.

Native bundles under/tmp/pseudo-mnist-results:

- completion-ep/20261115T213655.682485Z-feddd8ca-917c5565
- completion-bptt/20261115T213718.443897Z-460552e4-6d1c3e09
- permuted_classification-ep/20261115T213744.927476Z-550a044a-4a660795
- permuted_classification-bptt/20261115T213806.465323Z-b0a87f93-ce4632c9

All four pass`ebl runs inspect --verify-artifacts --require-complete`. Durable
archive including initial/final optimizer checkpoints and readiness bundles:
results/pseudo-mnist/20261115/native.tar.gz,3,169,061bytes,SHA256
f61afd214e1113edee52ff80ddc3f9112b5445ee59104ef33a250ea563a87d7f.
Extract under/tmp to restore original locations. Small aggregate JSON is beside
the archive. The trained sources remain in the published launch commit.

Plot command:`python -m experiments.plot_pseudo_mnist --root /tmp/pseudo-mnist-results --output /tmp/pseudo-mnist-figures`.
Shared record/plots:/home/filip/analog-research/on-chip-learning/explorations/pseudo-mnist-20261115/.
Owning issue:https://github.com/filip137/on-chip-learning/issues/51.
Preparation and token usage not instrumented; raw evidence retained, no pruning.

Limit: this is a conditional ideal attention aggregator using independently
anchored value states per query, with external digital normalized scores and
digital surrounding BP/Adam. Block-GS iterations are not physical RC time.
It establishes neither full analog EP, transistor robustness, recurrence benefit
nor hardware speed/energy savings. No subsequent run is authorized by this
review alone; possible next work is the same32-step model on more seeds or
larger token counts, or a dedicated16-step training comparison.
