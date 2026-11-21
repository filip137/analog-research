# Supervised adaptation of faulty XB6 and XB12

The assigned S7 comparison (coordination issue 15) uses OPT-125M std3 at two depths,
two-sided unit-gain mapping, CMO 2% endpoint law, nominal or 3% permanent random
faults uniformly stuck in 9–88.2 µS and one 5σ verify with partner programming.
There is no drift or separate read noise. Every method starts from a clone of the
same fault-aware deployment and calibration. Report untrained and calibrated holds.

For HellaSwag, BoolQ and PubMed compare digital rank-16 LoRA and global physical
Child-Tuning-D (1%,10%) offline and through the deployed chip, and full offline
fine-tuning followed by P&V. One Fisher pass over the common 1,024 training items
defines both nested masks. Train 128 updates, batch 32. Digital AdamW and bounded
physical Adam use zero decay, the Child writer accumulates up to a 1.12 µS gate
without final flush. P&V and incremental writes obey the actual leg mask. A fault
retains its programmed stuck value through all writes. Tune rates on development
281101 with an independently characterized proxy (281102); score first family
281201. Historical 271002 serves only for hold replay. Use separate training,
calibration, development and evaluation cohorts and report task and WikiText
retention metrics on the actual recipient chip. All routes use digital gradients.
Compare on/off pairs within family at 3% and subtract the corresponding contrast
at 0%. One chip is exploratory. Record transfer, writes, reads and peak wear.

Execution shares the coordinator's 24 RTX 5090 GPU-hour cap, including retries.
Native canaries and caps precede dispatch. XB6 and XB12 have a provisional 3 h
per host each (trex and loulou). Measured throughput determines additional
allocation and admitted matched cells.

Historical replay requires an exact physical hash and WikiText KL within 1e-6
of 0.0570604421 (XB6) or 0.2138088234 (XB12). Both hosts use the staged native
`launch/task_manifest.py` then the frozen `launch/worker.py --prepare` and
`launch/worker.py`. Worker verifies the source receipt and executable, plans,
dry-runs, executes, checks the final bundle and emits `runs/<unit>/complete.json`
or `failed-<job>.json`. Unit names use xb<depth>-<task>-r<rate>-<phase>-<tag>,
root `~/analog-research/s7-task-faults`. Development and evaluation each have
their own config and pinned inputs. Watch `metrics.jsonl`, campaign log and
the terminal receipt. XB owner checks at five-minute cadence with a hard cap of
three charged hours per host. Source task calibration is frozen. Fisher logical
batch 8, Child cosine schedule over 128 steps with no warmup or gradient clipping.
All runtime comparisons are approximate simulated operations.

First native RTX canary stopped on upstream HWA provenance before training
(14.88 s XB6, 13.02 s XB12). The source recipe now stays immutable and the
supervised stage declares its downstream chip. Source 463d5bb completed all
seven learner paths and four controls in the next canary. Regenerating the
historical V100 with an RTX CUDA seed produced different held tensors, as the
native allocator documents for different GPU architectures. The historical
gate now restores the hash-pinned V100 state and calls the original full
WikiText scorer at its native batch size. The independently drawn RTX state
remains the deployment for the supervised canary, with its separate hash.
Use `launch/task_lane.py` for the budgeted sequential screen, with XB6
HellaSwag 1% and XB12 BoolQ 10% first, each followed by its 0% twin.

The archived V100 readout also depends on scalar-division rounding. CPU replay
of the original stored conductances with the pre-rounded FP32 reciprocal of
79.2 reproduces the published hashes for both depths exactly (980be1f8…,
7c96f4b7…). The archived decoder uses this arithmetic and the contemporary
simulations use their native two-sided mapping. The native sparse run saves
both fixed masks bit-packed by leg/global logical index for every trained view,
with physical-tile/flat-index tie breaking and separate training/storage costs.

### Live handoff, 21 November 2026

Frozen runner `04c8f6f` on both hosts. Final regression suite 752 passed,
5 skipped. Latest canary (v4) ran all seven routes and four controls on both
RTX 5090s. The branch/case used the archived V100 hold only for replay and a
new RTX draw for the learner checks. The verified original held-state hashes
were exact. Measured versus published WikiText KL was 0.05706044280646 vs
0.05706044211984 at L6 and 0.21380882420961 vs 0.21380882339150 at L12.
Every sparse route had zero outside-mask write attempts, and the native bundles
contain hash-verified packed masks for all three views.

The XB6 `trex` lane (PID 538401) started its full HellaSwag 1% pair. XB12
`loulou` (PID 1383741) started its full BoolQ 10% pair. Both use the run root
above. Their first native canary receipts are
`runs/xb6-hellaswag-r03-canary-v4/complete.json` and
`runs/xb12-hellaswag-r03-canary-v4/complete.json` on the respective hosts.
The current receipt and budget surface are `lane-status.json`, `lane.pid` and
`lane.log`. Reconcile `remaining_coverage` at the cap. The shared study owner
supervises both lanes and collects compact verified results on the coordinator.
The 4-step canaries exercised sparse upload; at the declared learning rate no
online update reached the gate in four steps. The real runs test the accumulated
online writer under the complete 128-step schedule.
