"""One tiny coupled DRN using the actual calibration, selection, writer and transport."""
from dataclasses import replace
import copy

import pytest
import torch

from ebl.schema.opt_cmo_drn import TaskAdaptation
from methods import drn_task_adaptation as s7
from methods.programming.cmo import apply_mapping, program_rewritten
from networks.opt.physical import CmoDrnBank
from test_opt_cmo_drn import tiny_model, cpu_threads


def model_and_bank():
    model, _ = tiny_model()
    apply_mapping(model, 3.)
    _, _, state = program_rewritten(model, 71, 'random', .03, capture_state=True)
    return model, CmoDrnBank(state, 'cpu')


def inputs():
    tokens = torch.tensor([[1, 3, 4, 2], [4, 2, 3, 1]])
    return tokens, {'order': torch.tensor([[0, 1], [1, 0]]), 'splits':{'train': {'tokens':tokens}}}


def test_eligibility_nested_deterministic_masks(cpu_threads):
    model, bank = model_and_bank()
    tokens, bundle = inputs()
    s = TaskAdaptation(fisher_items=2, steps=2, calibration_items=2, microbatch=1)
    view = bank.view(model)
    rank = s7.rank_masks(model,bank,lambda ids:model(ids,view),bundle['splits']['train'],s,lambda *_:None)
    assert rank['eligible'] == sum(int((~a.evidence['verify_flags']).sum()) for a in bank.arrays)
    small, big = [s7.unpack_masks(rank,f,model) for f in (.01,.10)]
    assert all(not bool((m & ~n).any()) for m,n in zip(small,big))
    assert all(not bool((n & a.evidence['verify_flags']).any()) for n,a in zip(big,bank.arrays))
    s = replace(s, method='child', lr=.1)
    initial = [a.held.clone() for a in bank.arrays]
    _, report = s7.train(model, bank, lambda: bank.view(model), bundle, s, big, True, lambda *_:None)
    assert report['attempts_outside_mask'] == 0 and report['healthy_writes'] > 0
    for a,mask,old in zip(bank.arrays,big,initial):
        assert torch.equal(old[~mask],a.held[~mask])


@pytest.mark.parametrize('method',['lora','child','full'])
def test_offline_fit_deploys_on_real_bank(cpu_threads,method):
    model, bank = model_and_bank()
    receiver = copy.deepcopy(model)
    tokens,bundle = inputs()
    s = TaskAdaptation(method=method,lr=.01,steps=2,calibration_items=2,microbatch=1)
    view = bank.view(model)
    calibration = s7.output_calibration(model,view,tokens,'cpu',1)
    calibration.attach(model)
    masks = [torch.zeros_like(p,dtype=torch.bool) for p in model.parameter_groups()[0]] if method=='child' else None
    if masks:
        for m in masks:
            m.flatten()[::7] = True
    ad, _ = s7.train(model,bank,lambda:s7.proxy_view(model,bank),bundle,s,masks,False,lambda *_:None)
    fit = s7.export_fit(model,method,ad,masks)
    if ad: ad.detach_hooks()
    calibration.detach_hooks()
    initial = [a.held.clone() for a in bank.arrays]
    received, report = s7.transfer(receiver,bank,fit,method,41)
    if received: received.detach_hooks()
    if method=='child':
        for a,old,m in zip(bank.arrays,initial,masks):
            assert torch.equal(a.held[~m],old[~m])
    assert (bank.costs()['device_writes'] > 0) == (method!='lora')
    assert report['transferred_bytes'] > 0
    assert torch.isfinite(receiver(tokens,bank.view(receiver))).all()


def test_native_run_stages_and_mask_handoff(tmp_path, monkeypatch, cpu_threads):
    from dataclasses import fields
    import json
    from types import SimpleNamespace
    from ebl.runtime.opt_cmo_drn import task_adaptation as task
    from ebl.runtime.store import sha256_file
    from ebl.schema.opt_cmo_drn import Config, TaskConfig
    from tests.test_mapping_faults import configs, deploy_and_tune
    _, bank, _, source, tokens = deploy_and_tune(tmp_path,monkeypatch,3.)
    _, base = configs(3.)
    teacher = tmp_path/'teacher.pt'
    base = replace(base, stage='task_adaptation',teacher_sha256=sha256_file(teacher))
    monkeypatch.setitem(task.HWA_DIGESTS,6,sha256_file(source))
    monkeypatch.setenv('EBL_CORPUS_ROOT',str(tmp_path))
    monkeypatch.setenv('OPT_DRN_RUN_ID','run')
    for name in ('pubmed','wikitext'):
        value = {'schema':'s7-task.v1','task':name,'splits':{k:{'tokens':tokens} for k in ('train','development','test')},
                 'order':torch.tensor([[0,1],[2,3]])}
        torch.save(value,tmp_path/(name+'.pt'))
    setting = TaskAdaptation(task='pubmed',task_sha256=sha256_file(tmp_path/'pubmed.pt'),wiki_sha256=sha256_file(tmp_path/'wikitext.pt'),
                             steps=2,fisher_items=4,calibration_items=2,microbatch=1,eval_microbatch=1,limit=2)
    def invoke(action,information,method,**paths):
        name = '-'.join((action,information,method))
        s = replace(setting,action=action,method=method,information=information,lr=.1,fraction=.1)
        c = TaskConfig(**{f.name:getattr(base,f.name) for f in fields(Config)},task_adaptation=s)
        request = SimpleNamespace(spec=c,output_dir=tmp_path/name,teacher_weights=teacher,weights=source,
                                  device_state=None if information=='clean' else bank,command=('test',),
                                  task_fit=paths.get('fit'),task_masks=paths.get('masks'))
        assert task.run(request) == 0
        result = tmp_path/name/'run'
        assert (result/'result.json').exists()
        return result
    ctrl = json.loads((invoke('controls','recipient','lora')/'summary.json').read_text())
    assert set(ctrl['baselines']) == {'fp32_teacher','ideal_source','mapped_target','fault_aware_hold','calibrated_hold'}
    assert ctrl['endpoint']['retention']['teacher_kl'] >= 0
    for info in ('clean','recipient'):
        rank = invoke('rank',info,'child')/'checkpoints/task_masks.pt'
        train = invoke('fit' if info=='clean' else 'learn',info,'child',masks=rank)
        if info=='clean':
            fit = train/'checkpoints/task_fit.pt'
            endpoint = invoke('receive','recipient','child',fit=fit)
        else:
            endpoint = train
        row = json.loads((endpoint/'summary.json').read_text())
        assert row['attempts_outside_mask'] == 0 and row['bank_costs']['device_writes'] > 0
