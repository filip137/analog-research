# Experiment loop (test)

`loop.py` closes the experiment loop through GitHub. Agents review results and
propose the next experiment; you approve each launch by merging a PR.

```
authorized note on loop-test, no run  -> launch (envelope + clean-tree checks, runner snapshot)
run finished                          -> validity guards + fresh reviewer agent -> results PR
                                         (auto-merged when guards pass and verdict != invalid)
all results merged                    -> fresh proposer agent -> proposal PR
you merge the proposal PR             -> its experiment note launches on the next tick
```

Run it from this worktree with the Python that has the runner's dependencies:

```
/home/filip/miniconda3/envs/py312/bin/python loop/loop.py tick   # one pass
/home/filip/miniconda3/envs/py312/bin/python loop/loop.py run    # tick until only your steps remain
```

## Rules the script enforces

- Agents are fresh `claude -p` calls limited to Read/Grep/Glob, with no MCP
  servers and a per-call cost cap. They answer in a fixed JSON schema; the
  script writes every file.
- The reviewer gets the experiment note from the launch commit, not the
  current one. A note edited after launch fails a validity guard.
- A note proposed by the loop launches only after its proposal PR is merged by
  `LOOP_APPROVER`. The loop never merges proposal PRs.
- Configs must pass `envelope.json` both when proposed and again at launch.
  `max_experiments` forces a human checkpoint.
- Each run executes a snapshot of the runner, recorded with the launch commit
  and the note hash, from a clean worktree.

## Settings

| Variable | Default |
| --- | --- |
| `LOOP_BASE` | `loop-test` |
| `LOOP_RUNS` | `../loop-runs` (outside the repo) |
| `LOOP_APPROVER` | `filip137` |
| `LOOP_MODEL` | CLI default |
| `LOOP_AGENT_BUDGET_USD` | `1.00` per agent call |
| `LOOP_AUTOMERGE_RESULTS` | `1` |

## Known gaps in this test

- The loop pushes with your GitHub identity, so `mergedBy` cannot tell your
  merge from a script's merge, and GitHub sends you no notifications. A GitHub
  App or machine user fixes both.
- No cron yet; run `tick` or `run` by hand.
- A rejected (closed) proposal PR pauses the campaign; write the next note
  yourself with `authorized_by`.
