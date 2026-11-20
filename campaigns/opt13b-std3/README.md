# CMO crossbar sources for OPT-1.3B

Preparation under [study #68](https://github.com/filip137/on-chip-learning/issues/68). Use the pinned pretrained teacher for each depth. L12 and L24 use unit-gain, zero-margin two-sided devices. Both selection and deployment program each selection or held-out array from its own endpoint stream, then read the apparent conductance difference divided by 79.2 µS. The fault cases use native Gaussian endpoint programming. Record clean, mapped-target, fault-free and faulty KL. Standard HWA trains with Gaussian noise alone. Stuck-device simulation for an explicitly corruption-aware source uses the two-sided targets as well.

Generate the shared corpus once on the input host. Each split has its own seed. Save generated IDs directly so prompt tokens and sampled EOS tokens survive. Run from the pinned XB worktree with its interpreter. Set EBL_CORPUS_ROOT to the staged corpora directory and TEACHER to the pinned 1.3B pytorch_model.bin.

```bash
"$PY" -m campaigns.opt13b-std3.inputs generation-probe --teacher "$TEACHER" --output "$OUT/generation-probe.pt"
"$PY" -m campaigns.opt13b-std3.inputs canary-corpus --teacher "$TEACHER" --output "$EBL_CORPUS_ROOT/opt13b-synthetic-canary.pt"
"$PY" -m campaigns.opt13b-std3.inputs corpus --teacher "$TEACHER" --output "$EBL_CORPUS_ROOT/opt13b-synthetic-50m.pt"
"$PY" -m campaigns.opt13b-std3.inputs recovery --teacher "$TEACHER" --output "$EBL_CORPUS_ROOT/opt13b-recovery-samples8.pt"
```

Archive stdout with each artifact. Pin its file SHA-256 and all split or sample hashes before staging. Generate recovery targets once with FP32 OPT on the existing 1,024 WikiText training windows. Every recovery arm loads the same tensor. Budget generation from warmed tokens/s. The four-update canary uses a separately named 256-window-per-split corpus. Full training and screening share the complete synthetic cohort.

For each depth (12, 24), phase (canary, screen, production) and learning-rate factor, render the lifecycle and use the normal planner. Set ID to the rendered lifecycle_id. Use factor 1 for the canary, 0.1, 1, 3 for screening and the selected factor for production. The possible single edge extensions are 0.01 and 9.

```bash
"$PY" -m campaigns.opt13b-std3.configure --depth "$DEPTH" --phase "$PHASE" --factor "$FACTOR" --corpus-sha256 "$DIGEST" --output "$SPEC"
"$PY" -m ebl lifecycle check "$SPEC"
"$PY" -m ebl lifecycle plan "$SPEC" --teacher-weights "$TEACHER" --device cuda:0 --cpu-threads 8 --python "$PY" --output "$SRC/results/lifecycles/$ID/plan"
```

Production fits have 12,200 updates, a 98-update warmup and cosine decay to 20% of peak. The screen uses 512 updates and a four-update warmup. Keep the earlier checkpoint on ties. The 256-window development split, two independent selection arrays and equal weighting of healthy and 3% faulty arrays decide source selection. Production evaluates the trained source and pretrained control on fresh assignments 271002–271004, generated test and WikiText test. On healthy arrays the untreated 24-hour endpoint is reported at CMO ×1 and ×4. The program-and-verify residual RMS is in normalized weight units. Confirm that older mapping and precision configurations replay in the native suite.

Sampling synthetic teacher text and std3 per-output-channel noise follow Büchel et al. (2025). The update gradient uses noisy forward weights as in Rasch et al. (2023). The finite data budget, historically prompted sampling, teacher KL, split-rate Adam, cosine decay and programmed development selection are study choices. No teacher hidden-state cache. Storage reservations and the GPU request are in the shared study record. CPU checks are preparation evidence. Production requires the mapped GPU canary.
