"""Compact table of series003 exp-001 crossbar cells from collected native fault_tuning bundles.

usage: python campaigns/fault-recovery-paper/collect_mapping_faults.py <collected root> <out.csv> <out.json>
<collected root>/<unit>/ holds measurements.json and summary.json of each unit's fault_tuning bundle.
"""
import csv
import json
from pathlib import Path
import sys


def row(unit, run):
    rows = {r["label"]: r for r in json.loads((run / "measurements.json").read_text())}
    s = json.loads((run / "summary.json").read_text())
    model, source, mapping, rate = unit.split("-")
    out = {"unit": unit, "model": model, "source": source, "mapping": mapping, "rate_percent": int(rate[1:])}
    for key, label in (("naive", "naive_hold"), ("fa", "fault_aware_hold")):
        r = rows[label]
        held = {p["name"]: p["scores"]["teacher_kl"] for p in r["held_out"]}
        out.update({f"{key}_dev": r["final"]["teacher_kl"], f"{key}_wikitext": held["wikitext"],
                    f"{key}_generated": held["generated"]})
    d = s["deployment_records"]
    st = d["device_statistics"]
    rw = st["rewrite"]
    out.update(ideal_dev=d["source_ideal_teacher_kl"], clean_mapped_dev=d["clean_mapped_readout_teacher_kl"],
               target_floor=st["target_at_floor_fraction"], target_max=st["target_at_max_fraction"],
               stuck=st["naive"]["stuck_fraction"],
               naive_below_floor=st["naive"]["below_floor_fraction"], naive_above_max=st["naive"]["above_max_fraction"],
               fa_below_floor=st["fault_aware"]["below_floor_fraction"],
               fa_above_max=st["fault_aware"]["above_max_fraction"],
               naive_rms=st["naive"]["apparent_residual_rms"], fa_rms=st["fault_aware"]["apparent_residual_rms"],
               flagged=rw["flagged_active"] + rw["flagged_reference"],
               rewritten=rw["programmed_active"] + rw["programmed_reference"],
               clipped_low=rw["clipped_targets_low"], clipped_high=rw["clipped_targets_high"],
               false_positive=rw["analysis"]["false_positive_devices"],
               false_negative=rw["analysis"]["false_negative_devices"], devices=st["devices"],
               gpu=s["gpu"], fa_apparent_sha256=s["apparent_sha256"]["fault_aware"])
    return out


def main(root, csv_path, json_path):
    rows = [row(d.name, d) for d in sorted(Path(root).iterdir()) if (d / "summary.json").exists()]
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(json_path).write_text(json.dumps(rows, indent=1) + "\n")
    print(len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:4])
