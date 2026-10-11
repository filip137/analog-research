import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'exp3_lora_followup', ROOT/'campaigns/opt-cmo-corruption/launch/exp3_lora.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def evidence(final=.08):
    report = {'learning_rate':.003, 'effects':{
        'development':{'deployment':{'teacher_kl':.13},'lora':{'teacher_kl':final}},
        'transfer_generated':{'lora':{'teacher_kl':99.}},
        'transfer_wikitext':{'lora':{'teacher_kl':99.}}}}
    selection = {'selected_learning_rate':.003, 'screen':[
        {'learning_rate':.0003,'development_kl':.1},
        {'learning_rate':.001,'development_kl':.09},
        {'learning_rate':.003,'development_kl':.08}]}
    return report, selection


def test_heldout_test_values_cannot_trigger_tuning():
    report, selection = evidence()
    a = MOD.followup_decision(report,selection,.02)
    report['effects']['transfer_generated']['lora']['teacher_kl'] = .00001
    report['effects']['transfer_wikitext']['lora']['teacher_kl'] = float('nan')
    assert MOD.followup_decision(report,selection,.02) == a
    assert not a['triggered']


def test_material_worsening_uses_best_lower_validated_rate():
    report, selection = evidence(.09)
    before = deepcopy(selection)
    out = MOD.followup_decision(report,selection,.02)
    assert out['triggered'] and out['learning_rate'] == .001
    assert selection == before
    report['learning_rate'] = .001
    assert not MOD.followup_decision(report,selection,.02)['triggered']


def test_tolerance_and_invalid_metric():
    report, selection = evidence(.081)
    assert not MOD.followup_decision(report,selection,.02)['triggered']
    report['effects']['development']['lora']['teacher_kl'] = float('inf')
    with pytest.raises(ValueError,match='Invalid development metric'):
        MOD.followup_decision(report,selection,.02)


def test_lora_gate_and_receipt_are_idempotent(tmp_path, monkeypatch):
    import json
    from campaigns.execution_service import digest
    root = tmp_path
    (root/'prepared/queue').mkdir(parents=True)
    (root/'readiness').mkdir()
    manifest = {'id':MOD.PHASE}
    (root/'prepared/queue/campaign.json').write_text(json.dumps(manifest))
    policy = {'version':1,'manifest_sha256':digest(manifest),
        'user_authorization':'if the loss worsens tune the lr','max_followups_per_rank':1,
        'relative_tolerance':.02,'deadline':MOD.DEADLINE}
    (root/'readiness/lr-followup-policy.json').write_text(json.dumps(policy))
    report, selection = evidence(.09)
    report['rank'] = 16
    selection.update(selected_job=MOD.PREFIX+'-r16-lr3em3-50m',
        progress_calibration={},phase_progress_calibration={})
    selection_path = root/'readiness/selection-r16.json'
    selection_path.write_text(json.dumps(selection))
    before = selection_path.read_bytes()
    calls = []
    monkeypatch.setattr(MOD,'service',lambda *args:calls.append(args))
    monkeypatch.setattr(MOD,'transition',lambda *args:calls.append(args))
    a = MOD.lr_followup(Path('/unused'),root,report,'r16-lr3em3-50m')
    frozen = (root/'readiness/lr-followup-r16.json').read_bytes()
    assert MOD.lr_followup(Path('/unused'),root,report,'r16-lr3em3-50m') == a
    assert (root/'readiness/lr-followup-r16.json').read_bytes() == frozen
    assert selection_path.read_bytes() == before
    assert [c[-1] for c in calls if c[1] == 'release'] == [MOD.PREFIX+'-r16-lr1em3-50m']*2


def test_dense_selection_requires_gain_and_uses_only_development():
    spec = importlib.util.spec_from_file_location(
        'exp4_dense_lr_test',ROOT/'campaigns/opt-cmo-corruption/launch/exp4_dense_lr.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    reports = [{'label':tag+'-dev','learning_rate':rate,'sequence_presentations':8192,
        'initial_development_kl':.13,'development_kl':.15,'full_seconds_with_margin':1800,
        'collection':str(rate),'effects':{'transfer_generated':{'teacher_kl':.001}}}
        for tag,rate in mod.RATES.items()]
    assert not mod.select(reports)['triggered']
    reports[3]['development_kl'] = .1
    reports[3]['effects']['transfer_generated']['teacher_kl'] = 999
    selected = mod.select(reports)
    assert selected['triggered'] and selected['learning_rate'] == .01
    reports[1]['development_kl'] = .1
    assert mod.select(reports)['learning_rate'] == .001
    with pytest.raises(ValueError,match='incomplete'):
        mod.select(reports[:-1])
