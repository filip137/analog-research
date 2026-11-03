"""ebl.lifecycle.programming_rows — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations


def flat_rows(rows, run_dir):
    return [{**{k: v for k, v in row.items() if k not in ('metrics', 'clipping')},
             **row['metrics'], 'run_dir': str(run_dir)} for row in rows]
