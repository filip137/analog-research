from dataclasses import replace
import pytest

from ebl.schema.opt_cmo_drn import Config
from ebl.lifecycle.tuning import screen_configs,refinement_configs,confirmation_configs,final_configs,choose,choose_tier


def rows(configs):
    return [{'id':name,'config':c,'score':1.+i*.1,'initial_score':5.} for i,(name,c) in enumerate(configs.items())]


def test_declared_ladder_and_restart_budgets():
    base=Config()
    screen=screen_configs(base,'full')
    assert len(screen)==8 and {c.training.max_updates for c in screen.values()}=={256}
    refine=refinement_configs(base,'full',rows(screen))
    assert len(refine)==8 and {c.training.max_updates for c in refine.values()}=={512}
    confirm=confirmation_configs(base,'full',rows(refine))
    assert len(confirm)==2 and {c.training.max_updates for c in confirm.values()}=={2048}
    final,verdict=final_configs(base,rows(confirm))
    assert verdict['verdict']=='confirmed' and set(final)=={'clean','std3'}
    for arm,c in final.items():
        assert c.training.arm==arm and c.training.epochs==5 and c.training.max_updates==0
        assert c.training.warmup_updates==245 and c.training.schedule=='cosine'
    assert replace(final['std3'],training=replace(final['std3'].training,arm='clean'))==final['clean']


def test_partial_coverage_and_no_improvement_never_start_long_runs():
    base=Config()
    with pytest.raises(ValueError):refinement_configs(base,'minimum',rows(screen_configs(base,'minimum'))[:-1])
    confirm=confirmation_configs(base,'minimum',rows(refinement_configs(base,'minimum',rows(screen_configs(base,'minimum')))))
    r=rows(confirm)
    with pytest.raises(ValueError):final_configs(base,r[:1])
    for row in r:row['initial_score']=0.5
    final,verdict=final_configs(base,r)
    assert final=={} and verdict['verdict']=='inconclusive'


def test_one_percent_tie_uses_smaller_rate():
    r=rows(screen_configs(Config(),'minimum'))[:2]
    r[0]['score']=1.005;r[1]['score']=1.
    assert choose(r)['config'].training.conductance_lr==1e-4
    r[0]['score']=1.02
    assert choose(r)['config'].training.conductance_lr==3e-4


def test_budget_includes_all_arms_selection_checkpoints_and_readiness():
    tier,estimates=choose_tier(update_seconds=1,selection_seconds=10,checkpoint_seconds=3,readiness_seconds=100)
    assert tier=='full' and estimates[tier]<19.2*3600
    tier,estimates=choose_tier(update_seconds=100,selection_seconds=10,checkpoint_seconds=3,readiness_seconds=100)
    assert tier is None and set(estimates)=={'full','medium','minimum'}
    with pytest.raises(ValueError):choose_tier(update_seconds=0,selection_seconds=1,checkpoint_seconds=1,readiness_seconds=1)


def test_same_tuning_plan_can_be_written_again(tmp_path):
    from ebl.lifecycle.tuning import write_configs
    configs=screen_configs(Config(),'full')
    write_configs(tmp_path,configs,{'phase':'screen'})
    write_configs(tmp_path,configs,{'phase':'screen'})
    changed=dict(configs)
    changed['screen-A-0']=replace(changed['screen-A-0'],training=replace(changed['screen-A-0'].training,digital_lr=2e-5))
    with pytest.raises(ValueError):write_configs(tmp_path,changed,{'phase':'screen'})
