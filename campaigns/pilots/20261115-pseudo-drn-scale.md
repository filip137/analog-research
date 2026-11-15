# Pseudoconductance attention + passive DRN on CIFAR-10

Assigned: add a DRN MLP after attention, try a harder task, examine attention
score computation. Extends #51. Local CIFAR-10, 64 color patch tokens (4x4x3),
width64, four heads, DRN hidden128 and differential output64. One hybrid block,
mean pooling and digital classification head. Replace the prior tokenwise FF
with a passive positive-conductance resistor/ideal-diode network, retaining
reciprocal hidden-output loading. Clamped dual-rail input, hidden >=0, free
output rails, unit ground shunts, fixed differential read gain32.

Both attention and DRN use 32 block-GS sweeps/phase. EP via +/- frozen output
error currents; ordinary BP through projections, score computation, normalizers,
residuals and classifier. Physical timing, PDK, noise/mismatch not modeled.
All model forwards use their actual capped equilibria. BPTT unrolls identical
updates; no feedforward-ReLU substitution in the DRN arms.

Four arms, one seed20261115, identical data/order and shared initial tensors:
DRN+dot/EP; DRN+dot/BPTT; DRN+negative-L1/EP; digital-MLP+dot/BPTT. Last arm is
an architecture control, not an equal-device-count DRN-gradient comparison.
Attention controls are softmax scores. Dot scores q.k/sqrt(d); distance scores
-sum(abs(q-k))/sqrt(d). Both score paths use digital BP for training; investigate
analog forward/backward operation counts and interface requirements separately.

45k train/5k stratified validation (500/class,seed4100), original10k test;
batch32, 2000 updates, Adam .003, FP32/noTF32. Train random crop with four-pixel
reflection padding and horizontal flip; fixed unaugmented eval; pixels[-1,1].
Initial/final fixed24-example gradient cohorts, caps8/16/32/64 and long128
reference qualified against256. Report cosine and relative L2 globally and by
group, KKT residual, accuracy/CE, uniform-attention replay, and branch ranges.
Agreement gate cosine>=.99 and relL2<=5%, report failures without retuning.
Positive learning result means full-test accuracy above initialized/majority
baseline; no target state-of-the-art score or EP superiority assumed.

Local RTX3090 verified idle with23GB free. Total <=60GPU minutes inclusive of
readiness/retries; <=900s per arm with shared-budget guard; <=500MB raw /tmp
results and durable compact backend archive. Source pinned after merge/push.
Current agent owns monitoring; native heartbeat every50updates and check
five-minute stall gate. No remote host/GPU-sharing/settings changes. Preserve
operational failures and numerical failures. At most one identical pre-update
CUDA startup retry per case; bounded implementation fixes during readiness.

## Readiness and launch

Five numerical tests pass: DRN EP/BPTT/finite differences on an active/inactive
diode mixture; the earlier attention checks; L1 sign backward and its composition
with the softmax Jacobian. Public two-update CLI smokes for DRN-dot-EP,
DRN-L1-EP and digital-dot-BPTT completed full evaluation and all gradient probes.
Worst smoke group relative error is0.313%/0.297%/0.0085%; all 128/256-step
references qualify. Raw readiness:/tmp/pseudo-drn-readiness; logs
/tmp/pds-smoke-<arm>.log. Conservative readiness charge300s.

DRN equations, with clamped u=[x,-x], positive G1/G2 and hidden diode h>=0:
D_h=1+sum_rows(G1)+sum_cols(G2); D_o=1+sum_rows(G2).
Each sweep h=max(0,(uG1+oG2^T)/D_h), o=(hG2-injected_error)/D_o.
Readout32*(o_plus-o_minus). G1=softplus(raw1)/128 and G2=softplus(raw2)/128;
these are actual ideal conductances for every forward, not surrogate weights.
Unit shunts preserve strict convexity. EP input error is lambda_h G1^T;
conductance errors are contrasts of squared physical voltage drops. DRN
max nudge-current amplitude1e-4 (scaled and divided back out), attention.005.
Diode active-set changes make finite-amplitude DRN EP approximate.

Source/config/notes commit and merge/push before evidence. Launch:
`python -m experiments.run_pseudo_drn_scale --output-root /tmp/pseudo-drn-results`
from its pinned source checkout, with Python env py312. Supervisor retains a
four-byte CUDA tensor/context across workers as established in previous pilots.
Native CLI: `python -m ebl train --config examples/pseudo_drn_scale/<arm>.json
--output-dir /tmp/pseudo-drn-results/<arm>`.
Current agent owns supervisor/log/metric monitoring and final collection;
check50-update heartbeats and terminal summary+artifact inspection. Budget
guard requires900s remaining before a new arm starts. No tuning after launch.

## Declared score follow-up (before execution)

The scorer analysis derived a distinct normalization-free candidate: negative
L1 scores imply s<=0, so feeding a=exp(s) directly gives0<a<=1 and the ideal
aggregator computes normalized sigmoid(s) weights plus its null anchor. This
eliminates the separate row softmax, but changes the operator and total control
current. The block-GS contraction remains below.5 because a²/(1+a)<=a/2.

Add one explicit DRN-l1_raw-EP arm after the four original arms finish. Question:
does this simpler control interface still learn CIFAR? Change only score-to-gate
normalization, matched against DRN-l1-EP; preserve initialization,data,2000
updates,32sweeps,Adam,batch size and all diagnostic criteria. Uniform ablation
preserves each actual row's total current. Original pinned four runs remain
unchanged. Reuse same shared60-minute/500MB allowance: first two full cases
consumed222.3s plus300s conservative readiness; >3000s remains before pending
two arms and the single additional <=900s arm. No extra host or model-size
sweep. New source/config committed and published before the follow-up launch.

## Collected evidence and review

All five full cases completed exactly 2,000 updates, plus four two-update
readiness cases. No operational failures or retries. Original source is
19c6cffb4177d6334e1a898a4d272e203e27408c; follow-up source is
a17c879b1dba950927db6f3d20024abaca9fe877. All full manifests report clean
source. Supervisors 3563631/session97544 and 3568864/session33422 exited;
no active cases remain. Source trees were /tmp/pseudo-drn-source and
/tmp/pseudo-drn-source-raw, removed at closeout along with the working tree.

| Case | Validation accuracy | Test accuracy | Test CE | Final worst gradient relative L2 |
|---|---:|---:|---:|---:|
| DRN/dot EP | 42.38% | 42.79% | 1.5551 | 0.0459% |
| DRN/dot BPTT | 42.38% | 42.74% | 1.5552 | 0.0457% |
| DRN/L1 EP | 42.90% | 44.50% | 1.5115 | 0.0345% |
| Digital/dot BPTT | 42.20% | 43.16% | 1.5341 | 0.0140% |
| DRN/direct-exp L1 EP | 43.40% | 45.31% | 1.5011 | 0.0533% |

Gradient column takes the worst non-negligible group over EP/capped-BPTT/long
reference comparisons on the final 24-example cohort at32sweeps. All initial
and final gates passed, with qualified128/256-step references. Worst initial
relative error was0.389%, minimum cosine0.99999249; final minimum cosine
0.999999858. Six numerical tests pass. Final8-sweep probes also pass, but no
8-sweep training run was made. DRN final fixed-point residuals are zero in
FP32; attention residuals at most2.48e-7. Conductances remain positive.

The initial256-example validation accuracy is9.77% for all arms. Final uniform
attention replays reduce same-cohort accuracies from41.02/41.80/46.88/38.67/43.36%
to19.53/19.53/22.66/25.00/21.48% respectively. This supports reliance on learned
controls, but is not a separately trained no-attention control. DRN arms have
56,960parameters; the digital architecture control has40,576. All shared
initial tensors, sampler RNG and augmentation RNG are bit-identical. All six
dataset hashes match. Evaluation covers the full5kvalidation and10ktest.

`ebl runs inspect --verify-artifacts --require-complete --json` verified every
explicit full-run bundle. Native paths are below /tmp/pseudo-drn-results/<case>:

```text
drn-dot-ep/20261115T221411.304929Z-87897028-a1acbffa
drn-dot-bptt/20261115T221550.184136Z-b189f864-5cd79f73
drn-l1-ep/20261115T221756.474715Z-5c4c7813-35f086b4
digital-dot-bptt/20261115T221939.423601Z-0687f303-7ebe5e89
drn-l1_raw-ep/20261115T222221.743091Z-7f9edca4-00153cb6
```

All verification measurements are in the archived `full/collection.json`;
each run was required to be the sole bundle in its declared case directory,
never selected by recency. Archive includes raw
checkpoints, config/data/source provenance, logs, gradients, public inspection,
readiness bundles, and follow-up launch/collection scripts:

`/home/filip/analog-research/drn/results/pseudo-drn-cifar/20261115/native.tar.gz`

7,437,164bytes; SHA256
`ca118665e2f67515ec00e1803e9ea6e7f41a9de410bbc035031d891e70dd9ead`.
The archive was fully read and the durable copy hash checked. Conservative
budget charge862.07seconds/14.37minutes of60minutes, including300seconds
readiness plus the follow-up smoke; raw native evidence under10MB of500MB.

Verdict: the original learning and gradient-agreement criteria are met. The
passive DRN trains on CIFAR with local EP; matched dot EP/BPTT results are
nearly identical. L1 score controls learn, and removal of their separate
softmax shows no accuracy penalty in this pilot. This is one seed, one shallow
block and64k sampled training presentations; no general superiority claim.
Buffers, projection/score BP, normalization, fixed gain and Adam remain
digital/ideal. No measured-device training, physical energy or latency result.

The simpler candidate is a=exp(-L1/4), giving effective sigmoid-distance
weights through the existing aggregator. Its score backward needs a*r then
sign-routed query/key errors; it removes the softmax row-mean term, not all
products or interfaces. The next circuit question is a loaded differential
L1/exp cell with sign-routed adjoint and mismatch, with repeated-seed longer
training needed before claiming a score-accuracy advantage. These are
proposed next tests, not extra runs.

Shared [results and plot](https://github.com/filip137/on-chip-learning/blob/main/explorations/pseudo-drn-cifar-20261115/record.md)
and [score derivation with primary circuit sources](https://github.com/filip137/on-chip-learning/blob/main/explorations/pseudo-drn-cifar-20261115/attention-calculation.md).
Reproduce the plot using `python -m experiments.plot_pseudo_drn_scale --root
/tmp/pseudo-drn-results --output <directory>`. Tracking: [issue52](https://github.com/filip137/on-chip-learning/issues/52).
