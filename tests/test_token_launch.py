import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
LAUNCH=ROOT/'campaigns/drift-regimes/launch'
sys.path.insert(0,str(LAUNCH))
from make_token_manifest import manifest
from select_token_rates import select, MODES, RATES


def test_manifest_selects_only_one_evaluation_rate_per_mode():
    rates={m:3e-3 for m in MODES}
    m=manifest('/source','/out',rates)
    assert len(m['configs'])==8
    for seed in (271001,271002,271003):
        paths=m['configs']['tb-eval-'+str(seed)]
        assert len(paths)==5
        assert all('-lr0.003' in p or '-eval-reference' in p for p in paths)
    with pytest.raises(ValueError): manifest('/source','/out',{'random:32':.1})


def make_results(root):
    for mode in MODES:
        slug=mode.replace(':','-').replace('_','-')
        folder=root/('tb-tune-'+slug);folder.mkdir()
        arms={'T-samp8-lr0.01':{'mean_teacher_kl':.6914967786}}
        arms.update({'TB-'+slug+'-lr'+format(lr,'g'):{'mean_teacher_kl':score}
                     for lr,score in zip(RATES,(.9,.8,1.))})
        (folder/'complete.json').write_text(json.dumps({'trajectories':1}))
        (folder/'summaries.json').write_text(json.dumps([{'realization':271004,'replay_exact':True,
                                                         'source_unchanged':True,'arms':arms}]))


def test_rate_selector_uses_development_and_fails_on_changed_gate(tmp_path):
    make_results(tmp_path)
    assert select(tmp_path)['selected']=={m:3e-3 for m in MODES}
    p=tmp_path/'tb-tune-random-32/summaries.json'
    data=json.loads(p.read_text());data[0]['arms']['T-samp8-lr0.01']['mean_teacher_kl']=.8
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='reference mismatch'): select(tmp_path)
