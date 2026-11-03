---
id: "exp-004"
title: "Focused cap-128 HWA endpoint prerequisite"
status: "imported"
hypotheses: ["H-004"]
---

# exp-004 — Focused cap-128 HWA endpoint prerequisite

## Question and provenance

Focused cap-128 HWA endpoint prerequisite. A lower-from-RESET, half-step-tolerance, 128-pulse adaptive program-and-verify characterization can retain the endpoint, failure, corruption, saturation, and cost information required by the DRN HWA pilot while avoiding full target sweeps for the one-pulse calibration controller, the unused upper-start direction, and the deferred HfO2 stress control.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-004-hwa-device-model.md).

Dependencies and related controls: [Short-production programming and endpoint model](../results/exp-002-short-production.md), [128-pulse successor specification](../results/exp-003-cap128-successor.md).

## Frozen plans

- [Focused IBM OM cap-128 endpoint model for DRN HWA](../../../../../studies/ibm-reram-om-hwa-device-model-20260822-v1.json) — `ibm-reram-om-hwa-device-model-20260822-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_reram_short_production_review.md](../../../../../docs/ibm_reram_short_production_review.md).

<a id="source-ibm_reram_short_production_review-focused-prerequisite-for-the-first-hwa-network-pilot"></a>

## Focused prerequisite for the first HWA network pilot

The broader v3 matrix is still the matched OM/HfO2, lower/upper-start, and
controller comparison. The first DRN HWA pilot does not consume all of those
conditions, so its separate immutable prerequisite is
`studies/ibm-reram-om-hwa-device-model-20260822-v1.json` with profile
`hwa_production_cap128`.

That profile retains all 4,096 identity/repeat trajectories per target for
adaptive OM programming, but evaluates the one-pulse controller only on the
816 calibration trajectories. It uses the lower-from-RESET start only and
uses each identity's fixed sampled persistent lower/upper bounds for
analysis-only reachability. It therefore records 201,392 trajectories per arm
and 402,784 across the continuous and published-corruption OM arms: 6.67x
fewer trajectories than the broader v3 study. It does not provide evidence
about HfO2, upper-start behavior, or a full one-pulse endpoint model, and does
not rewrite the v3 contract.

After preparing the focused study, select it explicitly with
`local_short_launcher --study-profile hwa-om-cap128`; the launcher records the
two-arm 402,784-trajectory contract rather than inheriting the four-arm total.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
