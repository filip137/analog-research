# Mapping × faults on DRN6/DRN12 (evaluation only)

State: implemented, CPU-tested; canary pending on fifi. Branch `drn/20261119-mapping-faults`.
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

Pending launch.
