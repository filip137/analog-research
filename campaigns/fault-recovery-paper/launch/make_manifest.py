"""Jean Zay manifest of the fault-tuning screen units.

usage: make_manifest.py <JZ source dir> <JZ root> <out.json>
<JZ root>/inputs holds the corpora (symlinks), the pinned deployment and B scale, and (units a1-a3)
a symlink to the DRN half's stored V100 teacher-sample draw (checked by tensor hash in the stage).
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = "/lustre/fsn1/projects/rech/umg/ucy17uy"
PINS = json.loads((HERE.parent / "pins.json").read_text())


def main(source, root, out):
    inputs = f"{root}/inputs"
    manifest = {
        "schema": "fault_tuning.jz_manifest.v1",
        "source": source,
        "output": f"{root}/runs-v1",
        "corpus_root": inputs,
        "teacher": f"{P}/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin",
        "teacher_sha256": "2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9",
        "weights": f"{P}/opt-drift-times-20261028/inputs/crossbar-std3.pt",
        "weights_sha256": "86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028",
        "pinned": {
            "wikitext-2-raw-v1": "33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01",
            "opt125m-synthetic-50m": "ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984",
            PINS["deployment"]["name"]: PINS["deployment"]["sha256"],
            PINS["b_scale"]["name"]: PINS["b_scale"]["sha256"],
        },
        "units": {u: f"campaigns/fault-recovery-paper/lifecycles/opt125m-cmo-l6-fault-paper-tune-{u}.json"
                  for u in ("j1", "j2", "j3", "a1", "a2", "a3")},
    }
    Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
