"""Pin the analog-LoRA B scale: per-output-channel max |B| of the bscale unit's r16 digital LoRA.

usage: make_b_scale.py RUN_DIR OUTPUT.pt   (RUN_DIR: the completed fault_tuning run of the bscale unit)
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from core.artifacts import sha256_file  # noqa: E402
from core.serialization import load, save  # noqa: E402
from methods.fault_tuning import b_scales_from_lora  # noqa: E402


def main(run, output):
    result = json.loads((run / "result.json").read_text())
    if result.get("status", "complete") not in ("complete", "completed"):
        raise SystemExit("Expected a completed run.")
    artifact = next(a for a in result["artifacts"] if a["kind"] == "fault_tuning_state.dlora-r16-3em3")
    path = run / artifact["path"]
    if sha256_file(path) != artifact["sha256"]:
        raise SystemExit("Changed LoRA state.")
    state = load(path)
    if state["rank"] != 16:
        raise SystemExit("Expected the r16 digital LoRA.")
    count = len({k.split(".")[0] for k in state["lora"]})
    scales = b_scales_from_lora(state["lora"], range(count))
    save(output, {"schema": "fault_tuning.b_scale.v1", "rank": 16, "b_scales": scales,
                  "rule": "per-output-channel max |B| of the r16 digital LoRA at 3e-3 (declared fallback)",
                  "source_run": str(run), "source_state_sha256": artifact["sha256"]})
    print(json.dumps({"output": str(output), "sha256": sha256_file(output),
                      "min": min(float(s.min()) for s in scales), "max": max(float(s.max()) for s in scales),
                      "median_per_matrix": [float(s.median()) for s in scales]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
