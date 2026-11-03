import pytest
from workflow.slurm_recovery_watch import classify


def data(state='PENDING', case_state='running', code=None):
    row={'state':state,'reason':'Resources'}
    if code is not None:row['exit_code']=code
    return {'scheduler':{'jobs':[{'parent':'100','canary':False,'indices':[0]}]},
            'allocations':{'100_0':row},'cases':{'cases/random01-noisy':{'state':case_state,
            'native_run':'/run','has_artifacts':True,'metric_bytes':42,'progress':[{'step':128}]}}}


RUN={'id':'test','case_index':0,'case_id':'random01-noisy'}


def test_pending_and_planned_checkpoint_wait_are_not_failures():
    assert classify(RUN,data())['state']=='queued'
    result=classify(RUN,data('COMPLETED','checkpointed','0:0'))
    assert result['state']=='queued' and not result['complete_evidence']


def test_completion_requires_scheduler_and_semantic_evidence():
    assert classify(RUN,data('COMPLETED','complete','0:0'))['complete_evidence']
    assert classify(RUN,data('COMPLETED','running','0:0'))['state']=='failed'
    assert classify(RUN,data('FAILED','complete','1:0'))['state']=='failed'


def test_missing_accounting_is_unknown_and_running_native_failure_is_reported():
    d=data();d['allocations']={};assert 'probe_error' in classify(RUN,d)
    d=data('RUNNING');d['cases']['cases/random01-noisy']['native_error']={'type':'FloatingPointError'}
    assert classify(RUN,d)['log_error']


def test_unset_optional_heartbeat_and_log_fields_are_safe():
    result=classify(RUN,data('RUNNING'))
    assert result['alive'] and result['heartbeat_at'] is None


def test_prepared_cases_wait_for_canary_without_claiming_submission():
    d=data();d['scheduler']['jobs'][0]['canary']=True
    result=classify(RUN,d)
    assert result['state']=='queued' and not result['complete_evidence']
    assert 'probe_error' not in result and not result.get('process')
    d['scheduler']['released']=True
    assert 'probe_error' in classify(RUN,d)


def test_canary_requires_semantic_receipt_and_successful_slurm_exit():
    run={'id':'canary','canary':True}
    d=data('COMPLETED','complete','0:0');d['scheduler']['jobs'][0]['canary']=True
    assert classify(run,d)['state']=='failed'
    d['canary_complete']=True
    assert classify(run,d)['complete_evidence']
    d['allocations']['100_0'].update(state='FAILED',exit_code='1:0')
    assert classify(run,d)['state']=='failed'


def test_completed_canary_requires_actionable_controller_handoff():
    run={'id':'controller','controller':True}
    d={'observed_at':100,'supervisor':{'state':'awaiting_gate','updated_at':'1970-01-01T00:01:30+00:00','pid':123},
       'scheduler':{},'canary_complete':True}
    result=classify(run,d)
    assert result['state']=='running' and result['alive'] and result['log_error']
    d['canary_complete']=False
    assert classify(run,d)['state']=='queued' and not classify(run,d)['log_error']
    d['canary_complete']=True;d['scheduler']['released']=True
    assert not classify(run,d)['log_error']
