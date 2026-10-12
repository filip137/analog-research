import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/'campaigns/opt-cmo-corruption/launch/exp014_context.py'
spec=importlib.util.spec_from_file_location('exp014_context',PATH)
context=importlib.util.module_from_spec(spec);spec.loader.exec_module(context)


def reports():
    return [{'label':label,'cap':context.parts(label)[3],'length':context.parts(label)[2],
             'development':{name:{'initial':1.,'final':(.55 if context.parts(label)[2]==64 else .5)}
                            for name in ('256','1024')}} for label in context.labels('screen')]


def test_selection_shortest_qualifying_length_without_test_metrics():
    assert context.selection(reports())=={'cap2':128,'cap4':128,'uncapped':128}
    with pytest.raises(ValueError,match='Incomplete'):context.selection(reports()[:-1])


def test_selection_rejects_no_gain_and_conflicting_contexts():
    rows=reports()
    for r in rows:
        r['development']['1024']['final']=1.01
    assert all(v is None for v in context.selection(rows).values())
    for r in rows:
        r['development']['256']['final']=.5 if r['length']==64 else .8
        r['development']['1024']['final']=.5 if r['length']==1024 else .8
    assert all(v is None for v in context.selection(rows).values())


def test_selection_rejects_changed_initial_deployment():
    rows=reports();rows[0]['development']['256']['initial']=1.1
    with pytest.raises(ValueError,match='baseline'):context.selection(rows)
