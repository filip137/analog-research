---
experiment: "exp-002"
evidence: "imported-summary"
summary: "The archived review reports controller tradeoffs and rejects a blanket Gaussian approximation."
verdicts: {"H-002": "inconclusive"}
---

# Result: exp-002 — Short-production programming and endpoint model

[Experiment and historical specification](../experiments/exp-002-short-production.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| `results/ibm-reram-program-verify-noise-20260821-v2` — unavailable locally | Unavailable locally | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

The archived review reports controller tradeoffs and rejects a blanket Gaussian approximation.


<a id="source-ibm_reram_program_verify_noise_model-exploratory-compact-model-check-on-completed-v2-data"></a>

### Exploratory compact-model check on completed v2 data

The bounded candidate was selected after v2 completed, so this check does not
rewrite the v2 protocol and remains exploratory until the v3 rerun. Re-fitting
the completed fit identities and evaluating the untouched validation
identities gave:

| Compact model | Adequate conditions |
| --- | ---: |
| Fourth-order Gaussian | 0 / 16 |
| One global `Uniform(-tau,+tau)` | 7 / 16 |
| Target-conditioned eight-bin piecewise uniform | 16 / 16 |

For the piecewise-uniform model, pooled 90% coverage was `0.8978-0.9037`, 95%
coverage was `0.9488-0.9511`, and normalized median target-binned Wasserstein
distance was `0.0323-0.0479`. The single uniform principally failed the
one-pulse conditions because it cannot represent target-dependent mass within
the verify window.

The same post-completion check validates the paired-bound reachability
probabilities. Mean absolute fit-versus-held-out class-probability error was
`0.0088-0.0153` across the four populations; the largest individual
target/class difference was `0.0531`. At the nominal endpoints, the held-out
non-corrupt populations were:

| Population | x=0 exact below lower | x=0 full window unreachable | x=1 exact above upper | x=1 full window unreachable |
| --- | ---: | ---: | ---: | ---: |
| OM continuous | 51.5% | 47.1% | 52.9% | 47.5% |
| OM corruption arm, non-corrupt subset | 52.8% | 48.3% | 51.7% | 47.8% |
| HfO2 continuous | 52.5% | 31.9% | 52.5% | 29.9% |
| HfO2 corruption arm, non-corrupt subset | 52.6% | 30.9% | 53.1% | 29.7% |

These are structural bound events, not corrupt devices. In particular, the
HfO2 one-pulse controller apparently accepted about 98-100% of the exact-
unreachable endpoint cases because of its large apparent write variation;
their persistent endpoints did not become equal to the targets.

<a id="source-ibm_reram_short_production_review-post-completion-compact-endpoint-model-check"></a>

## Post-completion compact endpoint-model check

This analysis was added after v2 reached completion. It therefore does not
change the immutable v2 hypothesis or promote a post-hoc choice to
confirmatory evidence; the 128-pulse v3 successor predeclares the selected
candidate.

The accepted apparent residual is forced by verification into
`[-tau,+tau]`. A single uniform density over that window reproduced central
coverage but missed systematic target-conditioned mass. It passed 7 of 16
preset/corruption/controller/start conditions. The previously declared
fourth-order Gaussian passed none. An eight-bin piecewise-uniform density fit
separately at each target, with a `0.5` pseudocount per bin, passed all 16:

| Compact model | Adequate conditions | Normalized median target-binned Wasserstein |
| --- | ---: | ---: |
| Fourth-order Gaussian | 0 / 16 | failed gate in every condition |
| Global `Uniform(-tau,+tau)` | 7 / 16 | up to `0.1359` |
| Target-conditioned eight-bin uniform | 16 / 16 | `0.0323-0.0479` |

The selected model's pooled 90% coverage was `0.8978-0.9037` and its 95%
coverage was `0.9488-0.9511`. Failed non-corrupt endpoints and corrupt/stuck
endpoints remain separate empirical branches; neither is used to broaden the
accepted uniform density. Targets outside `[0,1]` are rejected by default,
and corruption is an identity-level mask that must persist across writes.

Non-corrupt bound variation is a second, distinct mixture. Pairing each
identity/repeat's persistent lower- and upper-conditioned states yields three
exact-target classes: below the lower bound, inside the interval, and above
the upper bound. Their fit-to-validation probability MAE was `0.88-1.53`
percentage points across populations, with a worst target/class error of
`5.31` points. At `x=0` and `x=1`, approximately half the devices cannot reach
the exact target. The half-step verify window reduces the structurally
unreachable mass to roughly `47-48%` for OM and `30-32%` for HfO2. Apparent
acceptance is modeled conditionally and does not change that latent class.

Historical source: [ibm-reram-program-verify-noise-20260821-v2](../../../../../docs/experimental_manifest.md#ibm-reram-program-verify-noise-20260821-v2).

- **Outcome:** mixed
- **Interpretation:** The predeclared hypothesis has a mixed outcome. No HWA, Tiki-Taka, LoRA, or network-accuracy conclusion follows from this characterization.

## Interpretation and limitations

The original study combines OM/HfO2, corrupt/repaired populations and target directions. Its named result root is absent locally; preserve the archived evidence and distinguish its post-hoc prefix analysis from a cap-128 replay.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Recover the named characterization bundles before fresh auditing. Keep verify-read savings and endpoint-quality tradeoffs separate.
