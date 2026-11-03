---
experiment: "exp-008"
evidence: "imported-summary"
summary: "Affine retained-floor mapping preserves accuracy while strongly attenuating voltages."
verdicts: {}
---

# Result: exp-008 — Literal versus affine floor mechanisms

[Experiment and historical specification](../experiments/exp-008-floor-mechanism.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-cmo-floor-mitigation](../../../../../results/mnist-cmo-floor-mitigation) | complete: 2 | 2 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

Affine retained-floor mapping preserves accuracy while strongly attenuating voltages.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [affine_floor_probe/20260730T093742.131477Z-ee47b30f-a87c03b1](../../../../../results/mnist-cmo-floor-mitigation/affine_floor_probe/20260730T093742.131477Z-ee47b30f-a87c03b1/result.json) | complete | selected accuracy: 96.11%; terminal accuracy: 96.11% |
| [affine_floor_probe_v2/20260730T095125.044508Z-ee47b30f-7664fae3](../../../../../results/mnist-cmo-floor-mitigation/affine_floor_probe_v2/20260730T095125.044508Z-ee47b30f-7664fae3/result.json) | complete | selected accuracy: 96.11%; terminal accuracy: 96.11% |

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-affine-floor-mechanism-follow-up"></a>

## Affine floor-mechanism follow-up

The next controlled probe uses the same clean HWA checkpoint, device seed,
physical range, acceptance fit, retention time, and stochastic terms, but
programs

```text
G_target_uS = 9 + (88.199997 - 9) * G_drn
G_drn_realized = G_realized_uS / 88.199997
```

The DRN still sees the physical floor:

```text
G_drn_nominal = 0.1020408 + 0.8979592 * G_drn
```

Unlike literal mapping, this affine transformation preserves the ordering and
uniformly scales the differences between all clean physical targets. Neither
W1 nor W2 has a low/high target clip. Endpoint noise and the final physical
clamp still place `18.32%` of stored W1 and `11.80%` of stored W2 exactly at
the floor.

| Measurement | Accuracy |
| --- | ---: |
| Shared clean HWA checkpoint | 96.37% |
| Deterministic literal hard floor | 81.83% |
| Noisy literal first write / stored realization | 80.61% / 80.34% |
| Deterministic affine retained floor | 95.94% |
| Noisy affine first write / stored realization | 96.12% / 96.11% |

The finite floor still changes the DRN normalization and strongly attenuates
voltages. Stored affine hidden/output RMS is `0.1922/0.02469`, compared with
`0.8761/0.2188` clean. Nevertheless, preserving the small learned values
removes almost the entire accuracy loss in this perfect-diode checkpoint.
The literal collapse was therefore dominated by many-to-one clipping rather
than floor loading alone.

A uniform floor changes both the voltage-weighted numerator and the incident
conductance sum in each coordinate update. Ordinary MVM reference subtraction
addresses only the first. The deterministic oracle reaches `96.37%` only when
both terms are corrected and the bias is scaled consistently; this is an
equation-level active bound, not a circuit result. Full derivation, the
on/off-ratio sweep, selector ablation, and artifacts are in
[Finite conductance floors in DRNs](../../../../../docs/conductance_floor_mitigation.md#finite-conductance-floors-in-drns).

<a id="source-conductance_floor_mitigation-main-finding"></a>

## Main finding

The first CMO result mixed together two different effects:

1. every physical edge retained the measured `9 µS` minimum conductance;
2. the literal mapping collapsed every learned conductance below
   `9/88.199997 = 0.1020408` DRN units to the same value.

The second effect caused most of the observed accuracy loss. On the shared
MNIST HWA checkpoint, a noisy **affine** mapping retained the same physical
`9 µS` floor without deterministically merging the small physical targets and
obtained `96.12%` accuracy on the first write (`96.11%` in the stored second
realization), compared with `96.37%` before deployment. The literal hard-floor
mapping obtained `80.61%` on its first write and `80.34%` in its stored
realization.

This does not make the floor electrically harmless. Under the affine mapping,
hidden/output voltage RMS fell from `0.8761/0.2188` to
`0.1922/0.02469`, approximately `4.6x/8.9x` attenuation. Accuracy remained
high in this idealized perfect-diode model, but analog noise margin, diode
operating point, readout resolution, power, and settling still require
separate study.

<a id="source-conductance_floor_mitigation-controlled-mnist-results"></a>

## Controlled MNIST results

All rows use the same `[1568, 100, 20]` perfect-diode HWA checkpoint and the
canonical 10,000-example MNIST test split. The noisy CMO rows use device seed
`17`, the `9–88.199997 µS` window, the `0.2%` program-and-verify fit, and
one-second relaxation/read noise.

| Scenario | Accuracy | Hidden RMS | Output RMS |
| --- | ---: | ---: | ---: |
| Clean HWA | 96.37% | 0.8761 | 0.2188 |
| Deterministic literal hard floor | 81.83% | 0.1460 | 0.01644 |
| Noisy literal floor, stored realization | 80.34% | 0.1441 | 0.01604 |
| Deterministic affine retained floor | 95.94% | 0.1892 | 0.02461 |
| Noisy affine retained floor, stored realization | 96.11% | 0.1922 | 0.02469 |
| Exploratory selector mask, top `384/60` inputs per column | 94.93% | 1.7993 | 0.3537 |

In the literal case, `98.13%` of W1 and `88.10%` of W2 clean targets are
below the floor. The nominal floor supplies a median `97.82%` of the realized
hidden degree and `85.18%` of the realized output degree. In the affine case,
the floor still supplies a median `86.79%` and `68.22%` of those respective
degrees, yet the distinct learned values remain available to the computation.
The affine run's total `realized-clean` RMSE is about `0.101`, but that number
is dominated by the intentional affine offset. Relative to the ideal mapped
target, stochastic programming/relaxation/read RMSE is `0.00428` for W1 and
`0.00452` for W2. Endpoint noise followed by physical clamping still places
`18.32%` of stored W1 and `11.80%` of stored W2 values exactly at `G_min`;
affine calibration avoids deterministic target clipping, not stochastic
floor hits.

The deterministic on/off-ratio sweep makes the distinction explicit:

| `G_min/G_max` | Literal hard floor | Affine retained floor |
| ---: | ---: | ---: |
| 1% | 96.51% | 96.30% |
| 2% | 96.10% | 96.20% |
| 3% | 95.27% | 96.16% |
| 5% | 93.14% | 96.08% |
| 7.5% | 89.31% | 95.97% |
| 10.204% (CMO) | 81.83% | 95.94% |
| 15% | 60.84% | 95.87% |
| 20% | 49.39% | 95.83% |

For this checkpoint only, reducing the literal floor ratio to `2%` keeps the
deterministic loss within `0.27` percentage points. The affine mapping is far
less accuracy-sensitive over the tested range, while its voltage amplitude
continues to shrink as the floor grows.

<a id="source-conductance_floor_mitigation-literal-versus-affine-kl-divergence"></a>

## Literal-versus-affine KL divergence

There is no unique KL divergence for these mappings until the compared
distribution is specified. Two complementary measurements were therefore
made on the matched stored CMO realizations:

1. a per-example 10-class distribution obtained by applying softmax to the
   paired DRN output differences;
2. a smoothed histogram distribution over the realized conductances.

The paired voltage differences are not calibrated logits. Raw-softmax KL is
strongly affected by the mapping-dependent voltage scale, so a second
predictive measurement centers each example and divides by its class-score
RMS before softmax. This normalization preserves top-1 decisions but imposes
an explicit deterministic temperature.

| Predictive comparison, literal vs affine | Literal → affine KL | Affine → literal KL | Symmetric KL | JS divergence | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw-score softmax | `0.0000878` | `0.0000880` | `0.0000879` | `0.0000220` | 81.23% |
| Per-example RMS-normalized | `0.3311` | `0.2631` | `0.2971` | `0.06449` | 81.23% |

All divergences are means over 10,000 examples and use natural-log nats. The
near-zero raw value is misleading: both floor-heavy networks have small
output amplitudes and therefore produce softmax distributions close to
uniform. Scale-normalized KL reveals their different class-score shapes.
For comparison, clean versus affine has only `0.00651` mean symmetric
scale-normalized KL and `98.95%` top-1 agreement, whereas clean versus literal
has `0.31285` nats and `81.18%` agreement.

For the stored conductance tensors, a 128-bin Jeffreys-smoothed histogram over
`[0,1]` gives:

| Conductance distribution | Literal → affine KL | Affine → literal KL | Symmetric KL | JS divergence |
| --- | ---: | ---: | ---: | ---: |
| W1 | `0.5371` | `1.1006` | `0.8188` | `0.1560` |
| W2 | `0.6976` | `0.9716` | `0.8346` | `0.18275` |
| W1 and W2 combined | `0.5383` | `1.0932` | `0.8157` | `0.15614` |

Histogram KL depends on binning. Across 64, 128, and 256 bins, combined
symmetric KL ranges from `0.7219` to `0.8670` nats, while JS ranges from
`0.1468` to `0.1662`, so the qualitative separation is stable.

<a id="source-conductance_floor_mitigation-recommended-sequence"></a>

## Recommended sequence

1. Use `affine_floor` as the primary floor-retaining deployment mapping when
   the program-and-verify device can target its full analog window. Keep
   `literal_conductance` as an explicit absolute-scale/clipping stress test
   and `normalized_offset` as an explicit cancellation abstraction.
2. Train several bases from initialization through the affine physical
   mapping and endpoint noise. Track accuracy, hidden/output voltage
   distributions, conductance sums, residual-current tolerance, and estimated
   power rather than accuracy alone.
3. Repeat the floor-ratio sweep across independently trained bases with
   validation-selected settings and an untouched final test split.
4. Train structured masks or bounded fan-in on validation data, then realize
   masked edges as true open circuits rather than devices held at `G_min`.
5. Prototype full uniform-offset KCL cancellation with gain error, floor
   mismatch, read noise, finite output resistance, and stability checks.
   Each all-ones crossbar offset is rank one, but the complete KCL/Laplacian
   perturbation also contains diagonal degree loading and is not rank one.
   Compare it with denominator-only calibration and a dual-rail
   constant-sum topology.

<a id="source-conductance_floor_mitigation-reproduction-and-artifacts"></a>

## Reproduction and artifacts

The validated analysis utility is
[`labs/tools/analyze_mnist_conductance_floor.py`](../../../../../labs/tools/analyze_mnist_conductance_floor.py).
The main machine-readable result is the local ignored artifact
`results/mnist-cmo-floor-mitigation/analysis/replay-v5/summary.json`; the
floor-ratio and cancellation plots are in the same directory. The official
affine probe is the local ignored run
`results/mnist-cmo-floor-mitigation/affine_floor_probe_v2/20260730T095125.044508Z-ee47b30f-7664fae3/`.
The predictive and conductance divergence result is the local ignored
artifact
`results/mnist-cmo-floor-mitigation/analysis/mapping-kl-v1/summary.json`.
The gate, component, wide-floor, imbalance, and synthetic-readout controls
are in the local ignored artifact
`results/mnist-cmo-floor-mitigation/analysis/affine-floor-mechanism-v3/summary.json`;
its plots are in the same directory. The corresponding utility is
[`labs/tools/analyze_mnist_affine_floor_mechanism.py`](../../../../../labs/tools/analyze_mnist_affine_floor_mechanism.py).

```bash
python labs/tools/analyze_mnist_conductance_floor.py \
  --clean-weights <clean-hwa-weights.pt> \
  --literal-weights <literal-cmo-weights.pt> \
  --affine-weights <affine-cmo-weights.pt> \
  --output-dir results/mnist-cmo-floor-mitigation/analysis/<new-run>
```

```bash
python labs/tools/analyze_mnist_affine_floor_mechanism.py \
  --clean-weights <clean-hwa-weights.pt> \
  --affine-weights <affine-cmo-weights.pt> \
  --output-dir results/mnist-cmo-floor-mitigation/analysis/<new-run>
```

The study is exploratory: it uses one trained base/device seed, reports on
the canonical MNIST test split, and models active correction algebraically
rather than with a transistor-level circuit. The CMO endpoint equations come
from AIHWKit's model of the measured device data; the physical window and
program-and-verify context are described by
[Falcone et al.](https://arxiv.org/abs/2502.04524).

Historical source: [mnist-cmo-floor-mitigation](../../../../../docs/experimental_manifest.md#mnist-cmo-floor-mitigation).

- **Outcome:** mechanism resolved; affine mitigation positive
- **Interpretation:** Hard clipping of `98.13%` of W1 and `88.10%` of W2, rather than passive floor loading alone, caused most of the literal accuracy collapse. Ordinary MVM reference subtraction is not exact for a DRN because the floor also changes the coordinate-update denominator.

## Interpretation and limitations

Avoiding deterministic many-to-one clipping explains much of the accuracy improvement. Preserved classification does not demonstrate harmless analog margins, readout noise or global equality of hypothesis classes.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Preserve target distinctions and separately test margin/noise sensitivity; high ideal accuracy does not settle hardware robustness.
