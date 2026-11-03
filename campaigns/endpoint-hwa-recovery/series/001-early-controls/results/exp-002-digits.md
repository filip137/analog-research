---
experiment: "exp-002"
evidence: "imported-summary"
summary: "The archived digits control shows a small LoRA gain without an HWA deployment benefit."
verdicts: {}
---

# Result: exp-002 — Perfect-diode digits HWA and LoRA

[Experiment and historical specification](../experiments/exp-002-digits.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

This record uses the historical sources linked below; no new root under results is assigned.

Earlier ignored evidence stays under labs/cases; follow the exact paths in the frozen manifest entry. No legacy directory was moved into results.

## Observations

The archived digits control shows a small LoRA gain without an HWA deployment benefit.


Historical source: [perfect-diode-hwa-lora-digits](../../../../../docs/experimental_manifest.md#perfect-diode-hwa-lora-digits).

<details>

<summary>Historical record: perfect-diode-hwa-lora-digits</summary>


### [perfect-diode-hwa-lora-digits](../../../../../labs/cases/perfect_diode_hwa_lora_comparison)

**Perfect-diode FP32, HWA, ReRAM, and LoRA comparison**

- **Finished:** 2026-07-28
- **Outcome:** mixed
- **Interpretation:** LoRA produced a small selected-checkpoint improvement, but the chosen HWA perturbation did not improve deployment over the ordinary base and left little resolved recovery headroom.
- **Details:** [legacy location](../../../../../labs/cases/perfect_diode_hwa_lora_comparison)


</details>

## Interpretation and limitations

Little resolved headroom and selected-checkpoint comparisons limit the claim.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use a regime with resolved deployment headroom before ranking recovery methods.
