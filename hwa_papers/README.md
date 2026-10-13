# HWA reference papers

Consult these two papers whenever a decision about hardware-aware (HWA)
training is made. That covers the noise model, its magnitude and scaling,
weight clipping and mapping, corruption-aware masks, the objective and
distillation, training data and token budget, the learning-rate schedule and
checkpoint selection. The experiment note records where each choice follows a
paper and where it deviates, and why. See [AGENTS.md](../AGENTS.md) and the
[Explorer procedure](../.agents/skills/experiment-loop/references/explorer.md).

The PDFs are kept here locally. They are ignored by git (`*.pdf`), so fetch
them from arXiv if they are missing, then check the SHA-256 below.

| Local file | Paper | SHA-256 |
| --- | --- | --- |
| `rasch2023_hwa_training_aimc_arxiv2302.08469v1.pdf` | M. J. Rasch, C. Mackin, M. Le Gallo, …, A. Sebastian, V. Narayanan, "Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators", [arXiv:2302.08469](https://arxiv.org/abs/2302.08469) v1 (2023), IBM Research; later published in *Nature Communications* (2023) | `6067c2f90b6a330669cbb35e5aa81e67b9238437f9a48e376f438d4008fd80a3` |
| `buchel2025_analog_foundation_models_arxiv2505.09663v3.pdf` | J. Büchel, I. Chalas, G. Acampa, …, M. Le Gallo, A. Rahimi, A. Sebastian, "Analog Foundation Models", [arXiv:2505.09663](https://arxiv.org/abs/2505.09663) v3 (27 Oct 2025), NeurIPS 2025, IBM Research and ETH Zürich; code at <https://github.com/IBM/analog-foundation-models> | `d917ec07fe1b9e4514d3c3ecd93a2f762e7925779dec9f83f8374f581a5f7c64` |

Downloaded on 2026-10-10.

## What to consult them for

Rasch et al. 2023 is a general AIMC HWA recipe for CNNs, RNNs and transformers
with a PCM standard model, built on AIHWKit. See Methods, "AIMC
hardware-aware DNN training". Its HWA training:
- **Noise placement.** Injects weight noise in both the forward and the
  backward pass, redrawn once per minibatch and not applied to the reference
  weights.
- **Noise distribution.** Draws the noise from the expected programming-error
  distribution, which depends on conductance, rather than generic Gaussian.
- **Noise ramp.** Ramps the injected noise strength up linearly over several
  epochs.
- **Peripheral parameters.** Learns the input range and the per-column
  weight-to-conductance scales.
- **Clipping.** Clips weights to the mapped range after every update.
- **Distillation.** Optionally distils from the FP teacher (temperature 10,
  75 % distillation and 25 % label loss); this helps when the output dimension
  is large.
- **Checkpoint.** Selects the last checkpoint, not the best one, and averages
  repeated runs.

Section 2.6 relates weight distributions to MVM fidelity. Supplementary A.2
gives per-network hyper-parameters.

Büchel et al. 2025 (Analog Foundation Models) is an HWA recipe for LLMs
(Phi-3-mini, Llama-3.2-1B).
- **Section 3.1, noise.** Per-output-channel additive Gaussian weight noise,
  `γ_weight · max|W_j,:| · τ` with γ = 0.02–0.03 (eq. 3; the convention our
  CMO `output_channel_abs_max` HWA noise follows). Noise-free weights in the
  backward pass.
- **Section 3.1, clipping.** Weight clipping to α·std after each step
  (eq. 4, α = 3.0). For LLMs clipping gave more robustness than noise
  injection, and the two combined worked best.
- **Section 3.1 and appendix I, quantization.** Learned static input ranges
  (DAC) and globally static output ranges (ADC).
- **Appendix B, training data.** Distillation from the FP model on synthetic
  data sampled from it: pure softmax sampling from BOS, continuing past EOS
  (B.1). Gains up to about 20 B tokens (B.2). Synthetic data beats FineWeb,
  which still works with distillation (B.3). Distillation matters (B.4).
- **Appendix C, ablations.** Noise type and magnitude (C.2), clipping α
  (C.3) and output quantization (C.1).
- **Appendix I, schedule.** AdamW with a polynomial learning-rate schedule and
  warmup.
- **Appendices E–H.** Evaluation over many benchmarks with 10 noise seeds.

## Where this campaign's choices stand against them

Recorded for [exp-001](../campaigns/opt-cmo-corruption/series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md)
onward.
- **Noise convention.** We use Büchel's per-channel additive noise
  convention. The backward pass goes through the noisy weights, as in Rasch
  and AIHWKit, not Büchel.
- **Not implemented here:** α·std weight clipping, DAC/ADC quantization,
  noise ramp-up and learned input ranges.
- **Our evidence on two of their recommendations:**
  - a decaying learning rate matters
    ([exp-003](../campaigns/opt-cmo-corruption/series/001-hwa-robustness/results/exp-003.md)
    part B);
  - HWA specializes to the training text distribution
    ([exp-004](../campaigns/opt-cmo-corruption/series/001-hwa-robustness/results/exp-004.md)).
