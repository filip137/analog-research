"""Freeze the FP32-direct pilot lifecycle: no HWA, verify pair rewrite, then the exp-017 recipe.

The pilot note owns the question and decision; Jean Zay execution uses run_stages.slurm.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from exp16_pair_rewrite import HOLD_ARMS, RECOVERY_ARM, lifecycle as exp16_lifecycle
from slice_campaign import write_frozen
from workflow.lifecycle import parse_lifecycle

LIFECYCLE_ID = 'opt125m-cmo-l6-fp32-direct-onchip'
NOTE = 'campaigns/pilots/20261019-fp32-direct-onchip.md'
ARRAY = 271002


def lifecycle():
    doc = exp16_lifecycle('std3', 'random3').to_dict()
    doc['lifecycle_id'] = LIFECYCLE_ID
    doc['question'].update(
        text='Without HWA, does the exp-017 on-chip recipe after the verify pair rewrite recover the '
             'directly deployed FP32 teacher, and how does it compare with the cited std3 rows?',
        decision='Per condition and held-out domain on array 271002: sign of final KL versus rewrite hold, '
                 'then a cross-run comparison with exp-017 std3 rows; descriptive only. ' + NOTE)
    doc['hwa']['arms'] = [{'id': 'pretrained', 'method': 'none'}]
    doc['devices'].update(assignment_seeds=[ARRAY], selection_seeds=[])
    # The raw hold is the contract's required no-write control for the writing arms.
    doc['onchip']['arms'] = [a for a in HOLD_ARMS if a['id'] in ('none', 'pair_rewrite_verify')] + [RECOVERY_ARM]
    return parse_lifecycle(doc)


def main() -> int:
    lc = lifecycle()
    path = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(LIFECYCLE_ID + '.json')
    write_frozen(path, lc.to_dict())
    print(path, lc.digest())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
