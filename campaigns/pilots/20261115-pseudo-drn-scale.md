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
