# IBM OM device layers

The IBM OM ReRAM integration lives in `training/ibm_om/`. It is split into
layers that each do one job and are each selected from the config.
`training/ibm_reram_hwa.py` stays as a re-export facade, so every historical
import keeps working.

## Layers

| Layer | Module | Selected by | Input → output | Must not see |
|---|---|---|---|---|
| Assignment | `population.py`, `sampling.py` | `preset`, `assignment_seed`, `corruption_policy` | DRN bindings → fixed `IbmReramArrayPopulation` (one AIHWKit-sampled cell per DRN conductance) | anything downstream |
| Topology | `topology.py` | `dual_rail_layout_by_parameter` | binding keys/shapes → dual-rail quads or G+/G− pairs | mapping, programming |
| Characterization | `characterization.py` | `characterization` (`none`, `reset_reads`) | population, topology → RESET-relative baselines or nothing | mapping, programming |
| Mapping | `mappings/<name>.py` | `target_mapping` | clean fractions `q = (G − Gmin)/span`, cell bounds, baselines → `MappingResult` (targets, report, `program_mask`, `OutOfSupportContract`) | execution, controller, plant, RNG |
| Programming | `programming/{ideal,compact,pulse}.py` | `execution`, `initial_state`, `controller`, `tolerance_step_ratio`, `verify_tolerance_absolute`, `maximum_program_pulses`, `target_out_of_support`, `endpoint_policy` | `ProgramRequest` + `ProgrammingContext` + RNG stream → endpoint, report, deployment | mapping name, mapping report, topology |
| Readback | `readback.py` | `clamp_to_parameter_bounds` | endpoint → DRN conductances `G = Gmin + span·endpoint`, restored after each context | mapping, programming internals |
| Coordinate | `coordinates.py` | `device_coordinate` (`logical_reference_subtracted`, `raw_active_v1`) | state ↔ [0, 1] conversion, cell bounds, plant type | — |

`modifier.py` is the only module that runs the layers in order. It does
nothing per-mapping: for mapping-specific setup it calls
`MAPPINGS[target_mapping].validate_bindings` and `.validate_population`.

The mapping is not undone on readback: the DRN sees the programmed (mapped)
conductances, and the straight-through gradient is applied to the clean
master weights.

## Stored config and compatibility

`IbmReramHwaConfig` (`spec.py`) is the canonical stored form, written into
modifier states, deployment bundles and checkpoint metadata. Five fields
declare the layer choices that used to be inferred from `target_mapping`:

| field | `raw_active_p90_quad` | `shared_reset_relative_quad` | other mappings |
|---|---|---|---|
| `device_coordinate` | `raw_active_v1` | `logical_reference_subtracted` | `logical_reference_subtracted` |
| `initial_state` | `conditioned_lower_boundary` | `exact_reset_bound` | `exact_reset_bound` |
| `verify_tolerance_absolute` | `0.00791586015294392` | `null` (step-ratio tolerance) | `null` |
| `clamp_to_parameter_bounds` | `false` | `true` | `true` |
| `characterization` | `none` | `reset_reads` | `none` |

- **Absent fields** are filled from this table by `legacy_explicit_defaults`, the only place the old inference survives.
- **Declared fields** must match the table. No other combination is implemented yet.
- **The other per-mapping requirements are unchanged:**
  - raw-active uses `controller=one_pulse` and `endpoint_policy=preserve`, and the model conductance bounds must be [0, 1];
  - `mapped_target` runs only with raw-active;
  - raw-active never uses `compact_endpoint`.

Artifacts written before these fields existed still load:

- Modifier states moved to version 3; version 2 is loaded through `backfill_config_dict`.
- Resume, decomposition and pilot-launcher provenance checks run saved modifier parameters through `backfill_weight_modifier_metadata` / `backfill_config_dict`.
- Old deployment configs rebuild through the dataclass defaults.

Still cross-layer by design:

- **Calibration branch.** The compact endpoint model and the controller step estimator come from the `continuous` or `published_corruption` branch, chosen by `corruption_policy`.
- **Compact fallback labels.** These mapping-specific strings in compact reports are supplied by each mapping's `OutOfSupportContract`, so programming never names a mapping. Mappings without a fallback (`literal_global`, `raw_active_p90_quad`) carry the historical quad labels.
- **`forward_logit_gain`.** A model readout setting that passes through unchanged.

## Nested config sections

New config files may write the weight-modifier parameters as sections
(examples in `examples/mnist_relu_drn/nested/`):

```json
"parameters": {
  "assignment": {"preset": "reram_array_om", "assignment_seed": 84001, "corruption_policy": "counterfactual_repaired"},
  "topology": {"dual_rail_layout_by_parameter": {"base.dense_weight.0": "halves", "base.dense_weight.1": "paired"}},
  "device_coordinate": "logical_reference_subtracted",
  "characterization": {"type": "reset_reads", "reset_read_samples": 8, "reset_guard_standard_errors": 3.0},
  "mapping": {"type": "shared_reset_relative_quad", "common_window_margin_fraction": 0.0,
              "reset_relative_mode": "continuous", "reset_relative_contrast_step": 0.095849},
  "programming": {"execution": "compact_endpoint", "controller": "adaptive", "start_protocol": "lower_to_target",
                  "initial_state": "exact_reset_bound", "tolerance_step_ratio": 0.5,
                  "verify_tolerance_absolute": null, "maximum_program_pulses": 128,
                  "target_out_of_support": "error", "endpoint_policy": "clip_0_1"},
  "readback": {"clamp_to_parameter_bounds": true},
  "streams": {"endpoint_seed": 84002, "noisy_evaluation": false},
  "forward_logit_gain": 707.945784384138
}
```

- Every section and field must be declared. Mapping-specific and characterization-specific fields are the only optional ones.
- The parser flattens sections into the canonical flat dict before validation, so a nested file and its flat equivalent resolve to the same spec.
- Frozen studies pin existing example files by their raw-byte hash. Do not rewrite them; add nested files instead.

## Verifying changes

`tests/test_ibm_om_golden.py` replays deterministic CPU goldens bit-exactly:
every mapping × execution cell, mapper and preflight behaviour, the config
corpus and validation grids, plant RNG order, STE gradients through the
MNIST runtime loop, and every recovery method including resume. The
populations carry non-zero cycle and write noise, so a change in RNG
consumption shows up.

- A refactor must keep the goldens passing unchanged.
- An intended numerical change needs a deliberate re-capture: run `python tests/ibm_om_golden_cases.py capture` and record the reason in the commit.
- `tests/test_ibm_om_real_artifacts.py` loads a pre-refactor CPU study run when it is present locally.

## Open follow-ups

1. Decide whether readback should undo the mapping, so the DRN sees the logical weight.
2. Stop the common-window and raw-active mappers from reading the true sampled device bounds.
3. Enable new coordinate × controller × execution combinations now that they are declared and stored.
4. Split the recovery methods into strategy modules, and expose the fast-reset programming settings in the recovery config.
5. Fix or remove the stale differential-pair launcher deployment-config check (`ibm_om_differential_pair_pilot_launcher.py`, `dict(config) != _expected_modifier_parameters(...)`).
