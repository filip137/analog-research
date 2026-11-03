---
experiment: "exp-003"
evidence: "imported-summary"
summary: "The screen finds a CMO mapping mismatch that a positive-only branch does not repair."
verdicts: {}
---

# Result: exp-003 — Endpoint device screen and CMO correction direction

[Experiment and historical specification](../experiments/exp-003-device-screen.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

This record uses the historical sources linked below; no new root under results is assigned.

Earlier ignored evidence stays under labs/cases; follow the exact paths in the frozen manifest entry. No legacy directory was moved into results.

## Observations

The screen finds a CMO mapping mismatch that a positive-only branch does not repair.


Historical source: [measured-device-screen](../../../../../docs/experimental_manifest.md#measured-device-screen).

<details>

<summary>Historical record: measured-device-screen</summary>


### [measured-device-screen](../../../../../labs/cases/perfect_diode_hwa_lora_comparison/device_screen)

**Measured-device deployment severity screen**

- **Finished:** 2026-07-28
- **Outcome:** mixed
- **Interpretation:** PCM and HERMES did not materially enlarge the gap under this calibration. CMO produced a more severe but differently composed perturbation.
- **Details:** [legacy location](../../../../../labs/cases/perfect_diode_hwa_lora_comparison/device_screen)


</details>

Historical source: [cmo-range-mismatch-lora](../../../../../docs/experimental_manifest.md#cmo-range-mismatch-lora).

<details>

<summary>Historical record: cmo-range-mismatch-lora</summary>


### [cmo-range-mismatch-lora](../../../../../labs/cases/perfect_diode_hwa_lora_comparison/device_screen/cmo_hwa_deployed)

**CMO/HfOx range-mismatch and LoRA probe**

- **Finished:** 2026-07-28
- **Outcome:** negative
- **Interpretation:** The current branch can add conductance but cannot subtract the excess caused by the CMO floor. The result diagnoses a correction-direction mismatch rather than a general failure of low-rank recovery.
- **Details:** [legacy location 1](../../../../../labs/cases/perfect_diode_hwa_lora_comparison/device_screen/cmo_hwa_deployed) · [legacy location 2](../../../../../labs/cases/perfect_diode_hwa_lora_comparison/device_screen/cmo_lora_run)


</details>

## Interpretation and limitations

Failure to subtract excess conductance diagnoses the tested correction direction, not all low-rank adaptation. Endpoint fits and floor conventions must remain explicit.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Separate correction direction and floor mapping before generalizing a low-rank failure.
