# Corrupted-device tuning completion, 15–16 November 2026

Status: prepared, not submitted. Owning issue: https://github.com/filip137/on-chip-learning/issues/33

Authoritative question, cases, scientific contract, budget, selection rule and handoff:
https://github.com/filip137/on-chip-learning/blob/main/campaigns/fault-recovery-paper/series/001-lr-tuning/experiments/exp-004-tuning-completion-20261115.md

The user requests preparation for tonight, following standing Stage-B Jean Zay
authority. Tune only at 3% on map 271001. Calibration uses only the same fixed
eight sampled teacher tokens (user, 15 Nov), with the shared recovery cohort,
300 Adam steps, batch 8, cosine decay and final 64-sequence dev-KL selection.
Per-channel gain/shift on complete MLP outputs before residual addition is
identity initialized; source parameters and held device states remain frozen.
Matched noiseless retains the noisy selected LR and gate, disabling only sampled
write noise. Historical dense noiseless remains a separate method.

Branch: `exp/20261115-fault-tuning-completion`. The source parent and generated configs preserve the
existing faulty deployments and historical method behavior. New control tests
check nominal gates, frozen physical states, identity initialization and replay.

Prepared ceiling: 24 GPU-hours total across both backends, including canaries,
retries and any documented edge checks; deadline 16 Nov 09:00 CET; 60 GB new
storage. Expected 6–9 GPU-hours, provisional until DRN12 timing. No new HWA,
0/1/5% tuning or paper cases. Preserve XB6's old analog-LoRA scale for its edge;
a scale change requires rechecking that screen. Calibration's five-point grid
is 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 for each model. Every matched control inherits
the noisy selected rate. Independent paper maps remain 271002–271004.

Before evidence launch: integrate/push this branch under the backend workflow,
stage a pinned source snapshot, run native plan/dry-run, and pass live Jean Zay
canaries with the exact resources/wrapper. Prepared configuration is not GPU
qualification. No job ID, live watcher or canary receipt exists at preparation.
The shared execution package records exact commands and current input staging.
A launch owner must arm the existing run-watch service, verify the first real
artifact and transfer the exact handles before leaving work unattended.

## Conditional boundary extension, 15 November

The four-arm digital extension completed as job247754. Rank64 final dev KL:
1e-4=0.0447313863 and3e-4=0.0443223974. Both make zero device writes;
the1%/lower-LR tie rule selects1e-4 at the lower boundary. Add exactly one
3e-5 rank64 case, changing only the lifecycle ID and arms relative to the
committed digital-edges lifecycle. All source weights, faulty deployment,
B-scale, lora seed, cohorts, eight teacher samples and300-step schedule stay
matched. Compare against the two completed rates and the original grid.

The new lifecycle is `fault-completion-20261115-xb6-digital-r64-lower-edge.json`.
It uses one V100 allocation with a30-minute cap as its own minimal one-case
live canary (the measured digital trajectory takes under one minute; source
and input verification dominates startup). No larger batch is released by
this run. Native planning/dry-run, static parsing, scheduler test-only and
post-run native artifact/coverage/identity gates apply. Code and wrapper are
unchanged. This adds0.5GPUh reserved wall time within the original24GPUh
ceiling and09:00CET deadline; it does not release paper arrays. Root decides
further boundary extensions, and the XB Sol medium agent monitors only.
