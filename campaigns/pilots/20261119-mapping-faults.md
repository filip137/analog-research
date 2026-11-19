# Mapping × faults on DRN6/DRN12 (evaluation only)

State: **complete** (25/25 cells verified on fifi, 19 Nov 2026); reviewed against the declared rule below. Branch `drn/20261119-mapping-faults`.
Authorization: user, 19 November 2026 (issue #33 of filip137/on-chip-learning); DRN share
of the joint 4 GPU-h budget is at most 1.5 RTX-5090 GPU-h including canaries and retries;
host fifi only (riri runs the #67 exp-002 queue).

Shared record (owns the question, factors, fixed choices 1–7, decision rule and handoff):
on-chip-learning [`campaigns/fault-recovery-paper/series/003-mapping-deployment/experiments/exp-001-mapping-faults-hwa.md`](https://github.com/filip137/on-chip-learning/blob/main/campaigns/fault-recovery-paper/series/003-mapping-deployment/experiments/exp-001-mapping-faults-hwa.md).

## Question (from the shared note; not changed here)

Which DRN deployment mapping (base, ratio2, ratio3: G′ = min(k·G, 88.2), bias current ×k)
gives the best deployed quality at 0/1/3/5 % random stuck devices (map 271002), for the HWA
sources DRN6 `dfb9f22f…` (CMO payload `86522e23…`) and DRN12 `6805b629…` (`ed53b6cb…`)?
No training. Naive (before partner rewrite) and fault-aware holds are scored on dev (64,
`7411cf50…`), WikiText-2 test (1,024, `5b51c74a…`) and generated (384, `bd00888a…`) KL against
the FP32 teacher `2d74da66…`. Primary endpoint: fault-aware WikiText-test KL at 3 % and 5 %.

## Implementation

| Semantic | Where |
| --- | --- |
| Optional `deployment.mapping_scale` (subclass `MappedDeployment`; legacy identities unchanged), range [1, 9.8], only for deploy/fault_tuning stages and never with `drift_tuning` | `hardware/deployment_config.py`, `ebl/schema/opt_cmo_drn.py` |
| `apply_mapping(model, k)`: k = 1 leaves the model untouched; otherwise the drift-tuning `remap` | `methods/programming/cmo.py` |
| Deploy: remap → `program_rewritten` (CMO draw → faults → 5σ verify → partner rewrite clamped to 9–88.2) ; `saved['mapping']`, `saved['current_scales']`, `deployment` dict carries `mapping_scale` | `ebl/runtime/opt_cmo_drn/recovery.py` |
| Deployment report: device counts (target/held/raw at ≥ 88.2 or ≤ 9 µS), receipt sums (faults, flagged, rewrites, clipped targets, node loading change) and per-layer median/p95/max of \|Δd\|/d for dh, do (`coefficients`) of held fault-aware (and naive) state vs mapped targets → `artifacts/deployment_report.json` | `methods/programming/cmo.py` `deployment_report` |
| fault_tuning restore: source-ideal KL, reapply mapping, mapped-target KL; mapping stats, bias currents and every programmed target must equal the saved deployment exactly | `ebl/runtime/opt_cmo_drn/fault_tuning.py` |
| Configs (paper-grid configs + `mapping_scale`, holds only) | `make_configs.py` → `examples/opt_cmo_drn/mapping_faults/` |
| Launcher (per cell: deploy, inspect, hold-only fault_tuning on that deployed.pt, inspect, verify) | `20261119-mapping-faults/run_lab.sh`, `verify_cell.py` |

Tests: `tests/test_mapping_faults.py` (15): legacy identity unchanged, scale recorded as float,
invalid scales and stages rejected, scale 1 leaves the model bit-identical and deploys
bit-identically to the legacy path (bank, receipt, dev KL, hold KLs), scale 3 equals the
drift-tuning remap followed by today's programming, restored holds score the ×3 bias
current, tampered scale/bias current/targets are rejected, report sanity (zero change on an
exact state). Neighbouring suites: 122 pass; `test_fault_paper_launcher.py::
test_exact_endpoint_coverage_and_native_canaries` fails identically on the trunk.

## Cases and budget

25 cells (one deploy + one hold-only fault_tuning bundle each): DRN6 and DRN12 × {base,
ratio2, ratio3} × {r00, r01, r03, r05} at map 271002, plus the check cell
`drn6-ratio3-r00-271001` (CMO exp-001 deployment: mapped 0.05273388 / release 0.05302006
on riri). Base cells declare `mapping_scale` 1.0 (new code path) and must replay the paper
grid: canary `drn6-base-r03-271002` FA-hold WikiText-test 0.18565395849884725 (riri), dev
0.18504243645013546. All cells on fifi (one RTX 5090); DRN12 paper deployments ran on fifi.
Measured paper-grid DRN12 timing on fifi: deploy 24 s, 34 s held-out per hold arm. Estimate
≈ 2 min per cell, ≈ 1 GPU-h total. Outputs: `fifi:~/analog-research/mapping-faults-20261119/runs-v1/<cell>{-deploy,}/run`;
compact collection `results/mapping-faults-20261119/`. Storage ≈ 0.3–0.6 GB per deployed.pt.

## Handoff

Launch commit `8577b1b` (merged, pushed); pinned checkout
`results/mapping-faults-20261119/source-8577b1b`; fifi staging `git archive` →
`~/analog-research/mapping-faults-20261119/source-drn-8577b1b` with `source_snapshot.json`
(5,112 files). Inputs `~/analog-research/mapping-faults-20261119/inputs` (paper-grid corpora,
samples, DRN12 source symlinked; DRN6 source copied from riri; all hashes match the pins).

```bash
bash $S/campaigns/pilots/20261119-mapping-faults/run_lab.sh $S $R/runs-v1 $R/inputs $WEIGHTS $TEACHER $LIMIT CELL...
```

- Canary `drn6-base-r03-271002` (tmux `mf33-canary`, 13:53–13:55 CET): rc 0, both bundles
  `complete valid=True`, `verify_cell` verified. Deployment held hashes equal the paper-grid
  riri deployment bit for bit; naive/FA dev, WikiText-test and generated KL equal the paper
  grid exactly (FA WikiText 0.18565395849884725). 75 s per DRN6 cell.
- Remaining cells: tmux `mf33-cells` on fifi from 13:55 CET, DRN6 (11 + check cell, limit
  900 s per call) then DRN12 (12, limit 1,200 s). Logs `launch/drn6-v1.log`, `launch/drn12-v1.log`
  (`rc=` line at the end). Owner: the executing agent, to completion.

Completion: all 24 cells plus the check cell finished rc 0 (DRN6 13:55–14:10, DRN12
14:10–14:33 CET). Every bundle `complete valid=True` under `runs inspect --verify-artifacts
--require-complete` and `verify_cell` verified. GPU time 0.64 RTX-5090 h (canary 74 s + 37.4 min).
Compact copies (no checkpoints) and `cells.csv`/`cells.json` (per-layer denominators):
`results/mapping-faults-20261119/fifi/`; deployed.pt files stay on fifi under `runs-v1/`.

## Evidence

Checks: every base cell replays the paper-grid map-271002 cell exactly (naive and FA dev,
WikiText-test and generated KL; held hashes identical to riri DRN6 and fifi DRN12 paper
deployments). Check cell `drn6-ratio3-r00-271001`: mapped targets 0.0527338775, FA release
0.0530200587 = CMO exp-001 riri (0.05273388 / 0.05302006); saturation 0.212 %.

Fault-aware WikiText-test KL (naive in brackets); 0 % row is also the clean deployed KL.

| Model | Rate | base | ratio2 | ratio3 |
| --- | --- | --- | --- | --- |
| DRN6 | 0 % | 0.0472 | 0.0484 | 0.0523 |
| DRN6 | 1 % | 0.0686 (1.83) | 0.0539 (0.128) | 0.0551 (0.073) |
| DRN6 | 3 % | 0.1857 (4.76) | 0.0772 (0.435) | 0.0646 (0.128) |
| DRN6 | 5 % | 0.3875 (5.19) | 0.1155 (0.982) | 0.0782 (0.197) |
| DRN12 | 0 % | 0.1920 | 0.2215 | 0.2792 |
| DRN12 | 1 % | 0.2137 (4.81) | 0.2265 (0.399) | 0.2854 (0.327) |
| DRN12 | 3 % | 0.4465 (5.41) | 0.2557 (1.98) | 0.3008 (0.509) |
| DRN12 | 5 % | 1.1044 (5.50) | 0.3201 (4.50) | 0.3278 (0.848) |

Median over layers of the per-layer median |Δdh|/dh (FA vs mapped targets) at 3 %: DRN6
0.183 / 0.062 / 0.022, DRN12 0.190 / 0.066 / 0.024 (base / ratio2 / ratio3); 5 %: 0.30 / 0.10
/ 0.036 and 0.31 / 0.11 / 0.040. The naive state changes denominators about half as much as
the FA state (the partner rewrite adds load). Clean mapped-target dev KL: DRN6 0.0477 /
0.0490 / 0.0527, DRN12 0.1981 / 0.2298 / 0.2883. Device statistics per cell are in `cells.csv`.

## Review (declared rule: lowest FA KL at 3 % and 5 % among mappings whose 0 % KL ≤ 1.10 × base)

- DRN6: ratio2 is admissible (0 % +2.6 %) and cuts fault damage (x % − 0 %) at 3 %/5 % from
  0.139/0.340 to 0.029/0.067. Ratio3 is lower still (damage 0.012/0.026) but its 0 % KL is
  +10.8 % (+10.9 % on the 271001 check), just outside the 10 % band, so the rule selects ratio2.
- DRN12: ratio2 (+15.4 %) and ratio3 (+45.4 %) both fail the 0 % criterion, so the rule keeps
  base, although ratio2 is far better at 3 % (0.256 vs 0.447) and 5 % (0.320 vs 1.104).
- Frozen expectation (ratio3 cuts damage several-fold; ratio2 between; denominator change
  shrinks accordingly) holds on both models. The 0 % cost grows with depth because the HWA
  sources were trained for base (no retraining). The user decides the rerun mapping.
