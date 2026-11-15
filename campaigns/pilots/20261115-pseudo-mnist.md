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
