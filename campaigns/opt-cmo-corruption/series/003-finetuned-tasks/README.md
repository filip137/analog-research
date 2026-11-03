# On-chip recovery of task-fine-tuned OPT-125M

In these experiments the deployed network is OPT-125M fine-tuned on a labelled
multiple-choice task (HellaSwag, SST-2 or PIQA) rather than the pretrained
language model. LoRA is merged into the FP32 weights before deployment; the
merged fc1/fc2 of the last six decoder MLPs are programmed on CMO arrays.
Every task also has a matched pretrained-teacher control that differs only in
the teacher.

- **Shared:** OPT-125M with analog fc1/fc2 in layers 6–11 (28,311,552 logical
  weights). The CMO 2% P&V law with active/reference pairs. The std3 Gaussian
  HWA recipe, trained on each task's own text. The verify partner rewrite as
  the standing default.
- **Not poolable:**
  - different tasks, teachers (fine-tuned or pretrained) or defect cases;
  - these rows and the WikiText/synthetic-corpus rows of series 001–002, which
    use another HWA corpus and budget.

- [exp-018: fine-tuned task recovery](experiments/exp-018-finetuned-task-recovery.md) (merged LoRA teachers; SST-2 complete, scope redirected)
- [exp-019: analog LoRA adapters vs base MLP arrays](experiments/exp-019-analog-lora.md): frozen std3 base, rank-16 adapters on crossbars, base-only vs adapter-only faults ([result](results/exp-019.md))
- [exp-020: on-chip recovery of the analog-LoRA deployments](experiments/exp-020-adapter-onchip-recovery.md): calibration vs adapter-only vs all-weight learning after the rewrite, up to 6% faults ([result](results/exp-020.md))

See the [campaign ledger](../../ledger.md).
