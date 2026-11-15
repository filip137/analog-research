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
