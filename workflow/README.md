# Simulation workflow

Start with the assigned experiment or pilot note and [AGENTS.md](../AGENTS.md).
This directory is documentation for navigating a simulation. The executable
interface is `python -m ebl`; see the [code ownership map](../docs/architecture.md).

| Step | Read | Result |
| --- | --- | --- |
| Define | [define.md](define.md) | Question, matched comparison, decision rule and budget |
| Prepare | [prepare.md](prepare.md) | Strict lifecycle, pinned inputs, bounded checks and execution handoff |
| Execute | [execute.md](execute.md) | Native bundles and observed progress within the authorized budget |
| Review | [review.md](review.md) | Verified coverage, measurements, interpretation and next decision |

The [execution policy](policy.md) owns process and authority. The
[experiment-loop skill](../.agents/skills/experiment-loop/SKILL.md) supplies the
procedures and git checkpoints. The [lifecycle contract](../docs/crossbar_lifecycle.md)
defines executable settings. Templates and earlier studies do not select scientific
settings for a new run.

`python -m workflow` was retired in the architecture migration. Use
`python -m ebl lifecycle` with the same lifecycle subcommands. Historical commands
and artifacts retain their recorded source checkout; their JSON schema identifiers
are unchanged.
