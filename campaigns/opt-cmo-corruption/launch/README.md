# Launch kit for the opt-cmo-corruption lifecycles

These are the scripts that ran Experiments 1–1d. Copy them to `<root>/launch/` on
each host. They live outside the clean source clone, so the clone stays clean;
`run_stages.py` refuses a dirty clone.
- **5090 and 3090 hosts:** `<root>` is `~/analog-research/opt-cmo-exp1/`.
- **Jean Zay:** `<root>` is
  `/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/`. `$SCRATCH` is not
  set in non-interactive SSH, so use the absolute path.

| File | Runs on | Does |
| --- | --- | --- |
| `run_stages.py` | any host | Runs named stages of a planned lifecycle through `python -m ebl train`. Inputs come from the plan, from completed upstream stages, or from `--input STAGE:ROLE=PATH`. A stage with a complete run is skipped |
| `train_stages.sh <id> <stage>…` | 5090/3090 hosts (`SRC` = clean clone) | Plans the lifecycle on that host, then runs the named stages, usually `prepare devices hwa__<arm>…` |
| `run_stages.slurm <id> [--input …] <stage>…` | Jean Zay (`sbatch --export=ALL,SRC=<clone>`) | The same on one V100, with `umg@v100`, `qos_gpu-dev`, `pytorch-gpu/py3/2.5.0` and `git/2.53.0` |
| `source_record.py <runs> <stage>` | training hosts | Prints `<source.pt path> <sha256>` of a completed HWA stage, from its run record |
| `inspect_collect.sh <id> <clone>` | Jean Zay | `ebl runs inspect --verify-artifacts --require-complete` on every bundle, then `workflow collect` |
| `pipeline.sh <arm>…` | the controlling machine | Per finished source: copies it to Jean Zay and verifies its hash, then submits its deployments (one dev job per arm, retrying while the submit limit is full). At the end: waits for every job, inspects and collects, inspects on the training host, copies rows and metrics to `results/lifecycles/<id>/`. Stops with `FAIL` on any unexpected state. See its header for the environment variables |
| `make_exp1e_lifecycle.py <rate>` | locally | Writes `opt125m-cmo-l6-exp1e.json` once Stage 1e-a has frozen the digital learning rate |

## Pitfalls seen so far

**Shells and clocks.**
- On Jean Zay, a non-interactive shell needs `. /etc/profile` before
  `module load`. Never source it under `set -u`: the script then dies
  silently. That happened once, in 1b collection.
- Compute nodes have no `git` without `module load git/2.53.0`.
- `nom-cool-1` has no tmux. Use `setsid nohup … &` and record the real PID by
  `pgrep -f`; `$!` inside `ssh "cd … && cmd &"` is not the training process.
- `riri`'s clock runs about 2 h slow, and `trex`'s is also off. Judge time
  windows by the controlling machine's clock.

**Hosts.**
- `nom-cool-2`: CUDA cannot initialize (`CUDA_ERROR_NOT_INITIALIZED`). Do not
  use it until it is rebooted.
- Do not probe `integnano2` (the Jean Zay jump host) with GPU commands.

**Scripts.**
- Never edit `pipeline.sh` while an instance runs: bash reads scripts
  incrementally. Copy it under a new name instead.
- `qos_gpu-dev` allows 10 submitted jobs per user and a 2 h wall time.
  `pipeline.sh` retries refused submissions for up to 2 h.

**Data.**
- You declined bulk copies of run bundles back to the local machine. Copy
  back only collected rows, inspect outputs and small `metrics.jsonl` files.

## Standing compute rules

From `docs/experiment_workflow.md`:
- **RTX 5090** (`trex`, `riri`, `fifi`, `loulou`):
  - 08:00–22:00 Paris time, at most two of our jobs at once;
  - 22:00–08:00, any idle 5090, but anything beyond two must finish by 08:00.
- **RTX 3090** (`nom-cool-1`; `nom-cool-2` is broken).
- **The idle rule:** only use a GPU that is idle when the job starts, and
  never run beside another process, including your other jobs.
- **Jean Zay:** `umg@v100`, preferably `qos_gpu-dev`.
