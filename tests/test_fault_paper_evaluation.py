"""CPU evidence that paper tests score the final live endpoint without training on it."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ebl.schema.base import ConfigError
from ebl.schema.opt_cmo_drn import parse_config
from ebl.runtime.opt_cmo_drn import fault_tuning as runner
from methods.config.fault_tuning import EvaluationCohort
from networks.opt.physical import CmoDrnBank
from tests.test_opt_cmo_drn import cpu_threads
from tests.test_opt_cmo_drn_fault_tuning import bank_and_model

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/opt_cmo_drn/fault_paper_completion/drn12-a-holds-noisy.json'


def raw_config():
    return json.loads(BASE.read_text())


def entry(tokens):
    return dict(name='paper_test', corpus='tiny', corpus_sha256='a'*64, split='test',
                start=1, sequences=len(tokens), cohort_sha256=runner.plain_hash(tokens))


def test_optional_field_preserves_legacy_resolved_identity():
    raw=raw_config(); parsed=parse_config(raw); old=asdict(parsed)
    assert parsed.fault_tuning.evaluation_cohorts == ()
    assert 'evaluation_cohorts' not in old['fault_tuning']
    raw['fault_tuning']['evaluation_cohorts']=[]
    new=asdict(parse_config(raw)); assert new['fault_tuning'].pop('evaluation_cohorts') == ()
    assert new == old


@pytest.mark.parametrize('mutate',[
    lambda e:e.update(extra=1), lambda e:e.update(start=-1), lambda e:e.update(start=True),
    lambda e:e.update(sequences=0), lambda e:e.update(cohort_sha256='wrong'),
    lambda e:e.update(name='../test'), lambda e:e.update(split=''),
])
def test_strict_evaluation_spec_rejection(mutate):
    raw=raw_config(); e=entry(torch.zeros((2,6),dtype=torch.int64));mutate(e)
    raw['fault_tuning']['evaluation_cohorts']=[e]
    with pytest.raises(ConfigError):parse_config(raw)


def test_duplicate_names_and_nonlist_rejected():
    raw=raw_config(); e=entry(torch.zeros((2,6),dtype=torch.int64))
    for entries in ([e,e],{'paper_test':e}):
        raw['fault_tuning']['evaluation_cohorts']=entries
        with pytest.raises(ConfigError):parse_config(raw)


@pytest.mark.parametrize('error',['hash','bounds','split'])
def test_cohorts_checked_before_learning(monkeypatch,error):
    tokens=torch.arange(24).reshape(4,6);e=entry(tokens[1:3])
    if error=='hash':e['cohort_sha256']='0'*64
    elif error=='bounds':e['start']=99
    else:e['split']='missing'
    settings=SimpleNamespace(evaluation_cohorts=(EvaluationCohort.parse(e),),sequence_length=6)
    monkeypatch.setattr(runner,'load_corpus',lambda *args:{'test':tokens.flatten()})
    with pytest.raises(ValueError):runner.evaluation_cohorts(settings)


class Store:
    def __init__(self,path):self.run_dir=path;path.mkdir();self.events=[];self.result=None
    def append_metric(self,row):self.events.append(row)
    def artifact_record(self,path,kind):return {'path':str(path),'kind':kind}
    def complete(self,**kwargs):self.result=kwargs
    def fail(self,error):self.error=error


def test_every_final_live_arm_scores_before_cleanup_without_test_selection(tmp_path,monkeypatch,cpu_threads):
    model,teacher,state=bank_and_model()
    train=torch.randint(0,17,(2,6));dev=torch.randint(0,17,(2,6));test=torch.randint(0,17,(2,6))
    raw=raw_config();raw['device']='cpu';s=raw['fault_tuning']
    s.update(sequence_length=6,recovery_sequences=2,fisher_batch=1,steps=2,batch=2,microbatch=1,
             development_every=1,evaluation_cohorts=[entry(test)])
    methods=['naive_hold','hold','noisy','matched_noiseless','lora','calibration']
    s['arms']=[dict(label=m,method=m,**({} if m in methods[:2] else dict(learning_rate=.1)),
                    **({'rank':2} if m=='lora' else {}))for m in methods]
    c=parse_config(raw);weights=tmp_path/'source.pt';torch.save({},weights)
    bank=CmoDrnBank(state,'cpu');physical={'schema':'opt_cmo_drn.deployment.v1',
        'source_weights_sha256':runner.sha256_file(weights),'deployment':asdict(c.deployment),
        'bank':state,'start_held_hashes':bank.held_hashes(),'programming':{}}
    physical_path=tmp_path/'deployment.pt';torch.save(physical,physical_path)
    store=Store(tmp_path/'run');seen=[];original_evaluate=runner.runtime.evaluate
    monkeypatch.setattr(runner,'cohorts',lambda settings:(train,dev))
    monkeypatch.setattr(runner,'load_samples',lambda *args:(torch.randint(0,17,(2,5,8)),weights))
    monkeypatch.setattr(runner,'load_corpus',lambda *args:{'test':torch.cat([torch.zeros(6,dtype=torch.int64),test.flatten()])})
    monkeypatch.setattr(runner.runtime,'configure_runtime',lambda *args:None)
    monkeypatch.setattr(runner.runtime,'teacher_model',lambda *args:teacher)
    monkeypatch.setattr(runner.runtime,'from_mapped',lambda *args:model)
    monkeypatch.setattr(runner,'_store',lambda *args:store)
    def evaluate(model,teacher,tokens,*args):
        if torch.equal(tokens,test):
            # Every test call follows its own final dev/updates, while digital hooks remain attached.
            seen.append({'hooks':bool(model.blocks[0]._forward_hooks),
                         'updates':sum(e.get('event')=='fault_tuning_update'for e in store.events)})
        return original_evaluate(model,teacher,tokens,*args)
    monkeypatch.setattr(runner.runtime,'evaluate',evaluate)
    request=SimpleNamespace(spec=c,weights=weights,teacher_weights=weights,device_state=physical_path)
    assert runner.run(request)==0
    rows=store.result['metrics']['arms'];assert [r['method']for r in rows]==methods
    assert len(seen)==6 and [x['hooks']for x in seen]==[False,False,False,False,True,True]
    assert [x['updates']for x in seen]==[0,0,2,4,6,8]
    assert not model.blocks[0]._forward_hooks
    for row in rows:
        assert set(row['held_out'])=={'paper_test'}
        assert row['held_out']['paper_test']['cohort']==entry(test)
        assert row['held_out']['paper_test']['metrics']['scored_tokens']==10
    assert all(r['updates']==2 for r in rows[2:])
    assert sum(e.get('event')=='fault_tuning_held_out'for e in store.events)==6
    assert all(r['curve'][-1]['step']==2 for r in rows[2:])


@pytest.mark.parametrize('mutation',['held','costs','rng'])
def test_scoring_rejects_physical_or_rng_mutation(monkeypatch,cpu_threads,mutation):
    model,teacher,state=bank_and_model();bank=CmoDrnBank(state,'cpu')
    tokens=torch.randint(0,17,(2,6));spec=EvaluationCohort.parse(entry(tokens));settings=SimpleNamespace(evaluation_microbatch=1)
    def bad_evaluate(*args):
        if mutation=='held':bank.arrays[0].held.add_(.1)
        elif mutation=='costs':bank.arrays[0].counters['write_attempts']+=1
        else:torch.rand(1,generator=bank.arrays[0].generator)
        return {'teacher_kl':0.}
    monkeypatch.setattr(runner.runtime,'evaluate',bad_evaluate)
    with pytest.raises(RuntimeError,match='mutated'):
        runner.score_held_out(model,teacher,bank,[(spec,tokens)],settings,'cpu',SimpleNamespace(append_metric=lambda row:None),'arm')


def test_run_rejects_bad_test_cohort_before_loading_targets_or_learner(monkeypatch):
    raw=raw_config();tokens=torch.zeros((2,256),dtype=torch.int64);e=entry(tokens)
    e['cohort_sha256']='0'*64
    raw['fault_tuning']['evaluation_cohorts']=[e]
    config=parse_config(raw)
    monkeypatch.setattr(runner.runtime,'configure_runtime',lambda *args:None)
    monkeypatch.setattr(runner,'cohorts',lambda *args:(tokens,tokens))
    monkeypatch.setattr(runner,'load_corpus',lambda *args:{'test':torch.zeros(3*256,dtype=torch.int64)})
    def forbidden(*args):raise AssertionError('Targets/learning reached before cohort verification')
    monkeypatch.setattr(runner,'load_samples',forbidden)
    request=SimpleNamespace(spec=config,weights=Path('unused'),teacher_weights=Path('unused'),device_state=Path('unused'))
    with pytest.raises(ValueError,match='cohort hash mismatch'):runner.run(request)
