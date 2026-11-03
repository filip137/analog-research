---
experiment: "exp-003"
evidence: "imported-summary"
summary: "Corrected logical-index MSE reaches 93.90–93.94%; loss and indexing interact."
verdicts: {}
---

# Result: exp-003 — RESET loss × bias × indexing factorial

[Experiment and historical specification](../experiments/exp-003-reset-factorial.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-relu-drn-reset-factorial-20260817-v1](../../../../../results/mnist-relu-drn-reset-factorial-20260817-v1) | running: 1 | 0 |
| [mnist-relu-drn-reset-factorial-local-shard-20260817-v3](../../../../../results/mnist-relu-drn-reset-factorial-local-shard-20260817-v3) | complete: 4 | 4 |
| [factorial_parallel_collected_20260817_v3](../../../../../results/factorial_parallel_collected_20260817_v3) | complete: 12 | 12 |
| [factorial_parallel_analysis_20260817_v3](../../../../../results/factorial_parallel_analysis_20260817_v3) | No native status records | 0 |
| [factorial_parallel_control_20260817_v3](../../../../../results/factorial_parallel_control_20260817_v3) | No native status records | 0 |
| [factorial-parity-smoke-local-v1](../../../../../results/factorial-parity-smoke-local-v1) | complete: 1 | 1 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [aggregate/results.json](../../../../../results/mnist-relu-drn-reset-factorial-20260817-v1/aggregate/results.json).

Existing analyses and receipts: [aggregate/results.json](../../../../../results/mnist-relu-drn-reset-factorial-local-shard-20260817-v3/aggregate/results.json).

## Observations

Corrected logical-index MSE reaches 93.90–93.94%; loss and indexing interact.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [stages/bias_free_logical_mse/local/bias_free_logical_mse_test/attempt-002/runs/20260817T061715.557043Z-fe9fe463-36936d4b](../../../../../results/mnist-relu-drn-reset-factorial-local-shard-20260817-v3/stages/bias_free_logical_mse/local/bias_free_logical_mse_test/attempt-002/runs/20260817T061715.557043Z-fe9fe463-36936d4b/result.json) | complete | evaluation accuracy: 93.90% |
| [stages/bias_free_logical_mse/local/bias_free_logical_mse_train/attempt-002/runs/20260817T054614.955891Z-9aae840c-5e0a5b79](../../../../../results/mnist-relu-drn-reset-factorial-local-shard-20260817-v3/stages/bias_free_logical_mse/local/bias_free_logical_mse_train/attempt-002/runs/20260817T054614.955891Z-9aae840c-5e0a5b79/result.json) | complete | selected student accuracy: 93.54%; initial student accuracy: 9.96%; terminal student accuracy: 93.42% |
| [stages/bias_legacy_kl/local/bias_legacy_kl_test/attempt-002/runs/20260817T061719.053297Z-1c3b6de4-fd3568b8](../../../../../results/mnist-relu-drn-reset-factorial-local-shard-20260817-v3/stages/bias_legacy_kl/local/bias_legacy_kl_test/attempt-002/runs/20260817T061719.053297Z-1c3b6de4-fd3568b8/result.json) | complete | evaluation accuracy: 80.45% |
| [stages/bias_legacy_kl/local/bias_legacy_kl_train/attempt-002/runs/20260817T060121.351224Z-9568c274-54a0aa58](../../../../../results/mnist-relu-drn-reset-factorial-local-shard-20260817-v3/stages/bias_legacy_kl/local/bias_legacy_kl_train/attempt-002/runs/20260817T060121.351224Z-9568c274-54a0aa58/result.json) | complete | selected student accuracy: 79.64%; initial student accuracy: 9.96%; terminal student accuracy: 60.62% |
| [akib/mnist-relu-drn-reset-factorial-akib-shard-20260817-v3/stages/bias_free_logical_kl/akib/bias_free_logical_kl_test/attempt-002/runs/20260817T061609.596212Z-9aed08f4-6fb31964](../../../../../results/factorial_parallel_collected_20260817_v3/akib/mnist-relu-drn-reset-factorial-akib-shard-20260817-v3/stages/bias_free_logical_kl/akib/bias_free_logical_kl_test/attempt-002/runs/20260817T061609.596212Z-9aed08f4-6fb31964/result.json) | complete | evaluation accuracy: 86.28% |
| [akib/mnist-relu-drn-reset-factorial-akib-shard-20260817-v3/stages/bias_free_logical_kl/akib/bias_free_logical_kl_train/attempt-002/runs/20260817T054623.063664Z-999352d7-345177f5](../../../../../results/factorial_parallel_collected_20260817_v3/akib/mnist-relu-drn-reset-factorial-akib-shard-20260817-v3/stages/bias_free_logical_kl/akib/bias_free_logical_kl_train/attempt-002/runs/20260817T054623.063664Z-999352d7-345177f5/result.json) | complete | selected student accuracy: 84.04%; initial student accuracy: 9.96%; terminal student accuracy: 78.56% |
| [akib/mnist-relu-drn-reset-factorial-akib-shard-20260817-v3/stages/bias_legacy_mse/akib/bias_legacy_mse_test/attempt-002/runs/20260817T061613.056264Z-6b67c92c-25395b4d](../../../../../results/factorial_parallel_collected_20260817_v3/akib/mnist-relu-drn-reset-factorial-akib-shard-20260817-v3/stages/bias_legacy_mse/akib/bias_legacy_mse_test/attempt-002/runs/20260817T061613.056264Z-6b67c92c-25395b4d/result.json) | complete | evaluation accuracy: 95.70% |
| [akib/mnist-relu-drn-reset-factorial-akib-shard-20260817-v3/stages/bias_legacy_mse/akib/bias_legacy_mse_train/attempt-002/runs/20260817T060050.675106Z-09e3be77-85e39bf5](../../../../../results/factorial_parallel_collected_20260817_v3/akib/mnist-relu-drn-reset-factorial-akib-shard-20260817-v3/stages/bias_legacy_mse/akib/bias_legacy_mse_train/attempt-002/runs/20260817T060050.675106Z-09e3be77-85e39bf5/result.json) | complete | selected student accuracy: 95.36%; initial student accuracy: 9.96%; terminal student accuracy: 95.04% |
| [trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_free_legacy_kl/trex/bias_free_legacy_kl_test/attempt-002/runs/20260817T065120.365165Z-def74cc9-a548402b](../../../../../results/factorial_parallel_collected_20260817_v3/trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_free_legacy_kl/trex/bias_free_legacy_kl_test/attempt-002/runs/20260817T065120.365165Z-def74cc9-a548402b/result.json) | complete | evaluation accuracy: 77.59% |
| [trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_free_legacy_kl/trex/bias_free_legacy_kl_train/attempt-002/runs/20260817T061841.253939Z-2be5492d-162041dd](../../../../../results/factorial_parallel_collected_20260817_v3/trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_free_legacy_kl/trex/bias_free_legacy_kl_train/attempt-002/runs/20260817T061841.253939Z-2be5492d-162041dd/result.json) | complete | selected student accuracy: 77.30%; initial student accuracy: 9.96%; terminal student accuracy: 54.48% |
| [trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_free_legacy_mse/trex/bias_free_legacy_mse_test/attempt-002/runs/20260817T065109.835165Z-28ec5dcb-91b8d28f](../../../../../results/factorial_parallel_collected_20260817_v3/trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_free_legacy_mse/trex/bias_free_legacy_mse_test/attempt-002/runs/20260817T065109.835165Z-28ec5dcb-91b8d28f/result.json) | complete | evaluation accuracy: 95.72% |
| [trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_free_legacy_mse/trex/bias_free_legacy_mse_train/attempt-002/runs/20260817T054619.051732Z-a1883e16-22406121](../../../../../results/factorial_parallel_collected_20260817_v3/trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_free_legacy_mse/trex/bias_free_legacy_mse_train/attempt-002/runs/20260817T054619.051732Z-a1883e16-22406121/result.json) | complete | selected student accuracy: 95.00%; initial student accuracy: 9.96%; terminal student accuracy: 94.54% |
| [trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_logical_kl/trex/bias_logical_kl_test/attempt-002/runs/20260817T065115.078859Z-d2df2dc3-c08bfc6b](../../../../../results/factorial_parallel_collected_20260817_v3/trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_logical_kl/trex/bias_logical_kl_test/attempt-002/runs/20260817T065115.078859Z-d2df2dc3-c08bfc6b/result.json) | complete | evaluation accuracy: 85.49% |
| [trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_logical_kl/trex/bias_logical_kl_train/attempt-002/runs/20260817T060221.117242Z-71699791-cafad11c](../../../../../results/factorial_parallel_collected_20260817_v3/trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_logical_kl/trex/bias_logical_kl_train/attempt-002/runs/20260817T060221.117242Z-71699791-cafad11c/result.json) | complete | selected student accuracy: 83.60%; initial student accuracy: 9.96%; terminal student accuracy: 75.84% |
| [trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_logical_mse/trex/bias_logical_mse_test/attempt-002/runs/20260817T065125.598662Z-c30019ea-a54c4b2d](../../../../../results/factorial_parallel_collected_20260817_v3/trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_logical_mse/trex/bias_logical_mse_test/attempt-002/runs/20260817T065125.598662Z-c30019ea-a54c4b2d/result.json) | complete | evaluation accuracy: 93.94% |
| [trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_logical_mse/trex/bias_logical_mse_train/attempt-002/runs/20260817T063436.028221Z-214f8082-ec2bf355](../../../../../results/factorial_parallel_collected_20260817_v3/trex/mnist-relu-drn-reset-factorial-trex-shard-20260817-v3/stages/bias_logical_mse/trex/bias_logical_mse_train/attempt-002/runs/20260817T063436.028221Z-214f8082-ec2bf355/result.json) | complete | selected student accuracy: 93.32%; initial student accuracy: 9.96%; terminal student accuracy: 93.30% |
| [20260817T054233.864886Z-5f1a7608-432ad40d](../../../../../results/factorial-parity-smoke-local-v1/20260817T054233.864886Z-5f1a7608-432ad40d/result.json) | complete | selected student accuracy: 12.50%; initial student accuracy: 18.75%; terminal student accuracy: 12.50% |

<a id="source-mnist_relu_drn_reset_factorial-results"></a>

## Results

All eight production runs and all eight fresh-process tests completed. The
analyzer verified the common teacher and device-source hashes, identical dense
device assignments, explicit minibatch input reset, one controlled production
restart per arm, the intended training indices, and fresh-process test indices
0->1 and 1->2.

| Hidden bias | Amplifier indexing | Paired MSE test accuracy | Teacher-KL test accuracy |
| --- | --- | ---: | ---: |
| off | logical | 93.90% | 86.28% |
| off | legacy process-global | 95.72% | 77.59% |
| on | logical | 93.94% | 85.49% |
| on | legacy process-global | 95.70% | 80.45% |

The factorial main effects on test accuracy are:

- teacher KL minus paired MSE: `-12.36` percentage points;
- legacy minus logical amplifier indexing: `-2.54` percentage points;
- bias enabled minus disabled: `+0.52` percentage points.

The indexing main effect must not be interpreted on its own because indexing
and loss interact strongly. Legacy indexing raises the paired-MSE result by
`+1.82` points without bias and `+1.76` points with bias, but lowers the KL
result by `-8.69` and `-5.04` points, respectively. Under corrected logical
indexing, switching from paired MSE to KL costs `7.62` points without bias and
`8.45` points with bias. Bias itself changes the logical paired-MSE result by
only `+0.04` points and the logical KL result by `-0.79` points.

Consequently, the historical `95.97%` number was modestly optimistic for the
corrected circuit: the matching legacy-index/MSE factorial cells reach
`95.70-95.72%`, whereas corrected logical indexing reaches `93.90-93.94%`.
The newer approximately `85.52%` KL result was not made artificially poor by
the indexing fix; the corrected logical KL cells reach `85.49-86.28%`. Most of
the old-versus-new gap is therefore attributable to the supervision objective,
with about `1.8` points of additional uplift from the archived indexing defect
in the MSE setting. Hidden bias is not a material explanation in this run.

The validated machine-readable output is under
`results/factorial_parallel_analysis_20260817_v3/`: `factorial_summary.json`,
`factorial_arms.csv`, and `factorial_accuracy.png`.

<a id="source-mnist_relu_drn_reset_factorial-interpretation-limits"></a>

## Interpretation Limits

This campaign has one independent run per cell. Its factorial contrasts
separate the three effects at seed 42 but do not estimate run-to-run
uncertainty. Any scientific claim about effect stability requires replication
with additional declared seeds. The old 95.97% result remains a historical
reference, not an additional factorial replicate, because its input-reset
semantics differ.

The complete sequential campaign manifest is
`campaigns/manifests/mnist_relu_drn_reset_factorial.json`. The executed
parallel shards use the same eight configs from artifact-recorded source
snapshot commit `1af0d2f6a7bf831d2d4a69f1ec3359963f4b0f76`, which is
not present in this checkout. The recorded source archive SHA-256 is
`d400859984eb550bac7cf1bc3e22b41c8f5ac171006c434fc9bec8c211c3910c`.
Their launch contract and cross-host parity result are local ignored
artifacts under
`results/factorial_parallel_control_20260817_v3/`. Collected raw output and
the generated JSON/CSV/PNG analysis are intentionally ignored by Git.

Historical source: [mnist-relu-drn-reset-factorial](../../../../../docs/experimental_manifest.md#mnist-relu-drn-reset-factorial).

- **Outcome:** historical result optimistic; attribution resolved, mechanisms open
- **Interpretation:** The original approximately `95.97%` result was too optimistic for the intended DRN: its matching legacy-index/MSE cells reach `95.70-95.72%`, while the corrected logical circuit reaches `93.90-93.94%`. The corrected logical KL result (`85.49-86.28%`) agrees with the newer run, so that result is representative of this KL objective rather than an indexing-induced underestimate.

## Interpretation and limitations

The factorial resolves the historical attribution at one seed. Legacy-index MSE gains roughly 1.8 points while legacy indexing harms KL. Bias is not the main explanation; the historical source is not an extra replicate.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Treat loss-by-indexing interaction as established for this seed; replication is needed before claiming a stable effect.
