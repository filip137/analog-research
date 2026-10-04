"""Count proposed partial-recovery budgets from the actual CIFAR topology.

This is a CPU architecture audit, not a training or recovery experiment.
No checkpoint, dataset, accuracy, write-cost or memory measurement is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix


def audit(classes: int, depth: int):
    suffix_name = {4: "last_two_blocks", 8: "last_four_blocks"}[depth]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        student = CrossbarSuffix(CifarResNet32(classes), suffix_name)
    student.enable_calibration(True)
    calibration = sum(p.numel() for p in student.calibration_parameters())
    total = student.q.numel()
    assert sum(spec.size for spec in student.layout) == total
    assert len(student.layout) == depth + 1
    assert calibration == depth * 2 * 64 + depth + 1 + classes
    layout = [
        dict(dataset=f"cifar{classes}", analog_convolutions=depth,
             matrix=spec.name, shape=list(spec.shape), offset=spec.offset,
             logical_weights=spec.size)
        for spec in student.layout
    ]
    rows = []

    def add(method, physical=0, digital=0, existing_calibration=0, selected=""):
        rows.append(dict(
            dataset=f"cifar{classes}", analog_convolutions=depth,
            method=method, deployed_logical_weights=total,
            physical_update_eligible_logical_weights=physical,
            physical_update_eligible_percent=100 * physical / total,
            new_digital_parameters=digital,
            new_digital_parameters_percent_of_base=100 * digital / total,
            existing_calibration_trainable_parameters=existing_calibration,
            total_trainable_parameters=physical + digital + existing_calibration,
            available_calibration_parameters=calibration,
            selected_matrices=selected,
        ))

    add("calibration_only", existing_calibration=calibration)
    groups = {
        "physical_classifier_only": [s for s in student.layout if s.name == "fc"],
        "physical_last_block_and_classifier": [
            s for s in student.layout if s.name.startswith("layer3.4.") or s.name == "fc"
        ],
        "physical_last_two_blocks_and_classifier": [
            s for s in student.layout
            if s.name.startswith(("layer3.3.", "layer3.4.")) or s.name == "fc"
        ],
        "physical_full_suffix": student.layout,
    }
    for name, specs in groups.items():
        add(name, physical=sum(s.size for s in specs),
            selected=";".join(s.name for s in specs))
    for percent in (1, 5, 10):
        # A budget only: no gradient-based or random mask is selected here.
        add(f"physical_selected_{percent}percent_budget", physical=total * percent // 100)
    classifier_size = next(s.size for s in student.layout if s.name == "fc")
    add("digital_dense_classifier_residual", digital=classifier_size, selected="fc")

    for rank in (2, 4, 8):
        parameters = 0
        for spec in student.layout:
            outputs, inputs = spec.shape[0], spec.size // spec.shape[0]
            assert rank <= min(inputs, outputs)
            # Flattened 3x3 convolution: A is rank x (64*3*3), B is 64 x rank.
            # Count an instantiated bias-free factor pair to check the formula.
            with torch.random.fork_rng(devices=[]):
                factors = nn.Sequential(nn.Linear(inputs, rank, bias=False),
                                        nn.Linear(rank, outputs, bias=False))
            count = sum(p.numel() for p in factors.parameters())
            assert count == rank * (inputs + outputs)
            parameters += count
        add(f"digital_lora_rank{rank}_all_matrices", digital=parameters,
            selected=";".join(s.name for s in student.layout))
    return rows, layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/cifar_partial_recovery_20260920")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, layout = [], []
    for classes in (10, 100):
        for depth in (4, 8):
            budgets, matrices = audit(classes, depth)
            rows.extend(budgets)
            layout.extend(matrices)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "parameter_budgets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "matrix_layouts.json").write_text(json.dumps(layout, indent=2) + "\n")
    notes = [
        "Counts come from the local CifarResNet32/CrossbarSuffix topology; no training ran.",
        "All arms retain the declared analog deployment depth and its physical faults.",
        "Except calibration_only, rows assume calibration is frozen after a common prefix.",
        "For joint calibration add available_calibration_parameters to the trainable count.",
        "Physical eligibility counts logical weights, not devices, actual writes or pulses.",
        "Digital residual counts exclude biases, held in the common calibration state.",
        "LoRA factors remain separate digital branches; merging can require dense writes.",
        "Sparse percentages are floored budgets; no sensitivity or random mask was fitted.",
        "Parameter counts do not measure activation memory, latency, energy or accuracy.",
        "The 4-convolution last-two-block arm equals full recovery by construction.",
    ]
    sources = [ROOT / "experiments/cifar_crossbar/model.py", Path(__file__).resolve()]
    manifest = dict(
        kind="static_architecture_audit", rows=len(rows), matrix_rows=len(layout),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
        torch_version=torch.__version__, notes=notes,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# CIFAR partial-recovery parameter budgets", "",
             "Static architecture audit; these are proposed methods, not recovery results.", ""]
    for classes in (10, 100):
        for depth in (4, 8):
            lines += [f"## CIFAR-{classes}, {depth} analog convolutions + classifier", "",
                      "| Method | Physical weights eligible (%) | Added digital parameters | Existing calibration trained |",
                      "|---|---:|---:|---:|"]
            for row in rows:
                if row["dataset"] == f"cifar{classes}" and row["analog_convolutions"] == depth:
                    lines.append(
                        f"| {row['method']} | {row['physical_update_eligible_logical_weights']:,} "
                        f"({row['physical_update_eligible_percent']:.3f}%) | "
                        f"{row['new_digital_parameters']:,} | "
                        f"{row['existing_calibration_trainable_parameters']:,} |"
                    )
            lines.append("")
    lines += ["## Interpretation", ""] + [f"- {note}" for note in notes] + [""]
    (args.output / "parameter_budgets.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "budget_rows": len(rows),
                      "matrix_rows": len(layout)}, indent=2))


if __name__ == "__main__":
    main()
