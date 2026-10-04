from copy import deepcopy
import pytest
from experiments.cifar_crossbar.sweep_config import SweepSpec,SOURCES,cases
from experiments.cifar_crossbar.hwa_fault_runtime import case_name
from experiments.cifar_crossbar.fault_config import METHODS
from experiments.cifar_crossbar.sweep_report import validate,aggregate


def result(backend='pcm'):
    rows=[]
    metric={'examples':10000,'accuracy_percent':80.,'teacher_kl':.5,'teacher_agreement_percent':85.}
    for source in SOURCES:
        for kind,rate in cases(SweepSpec(),source):
            case=case_name(kind,rate)
            for method in METHODS:
                curve=[]
                if method!='none':
                    for epoch in range(1,6):
                        cost={'array_reprogram_calls':0 if method=='calibration' else 704*epoch} if backend=='pcm' else {'max_recovery_cell_pulses':0 if method=='calibration' else 640}
                        curve.append({'epoch':epoch,'test':metric,'image_presentations':45000*epoch,'cost':cost})
                rows.append({'source':source,'case':case,'method':method,'p0_sha256':source+case,
                    'initial_apparent_sha256':source+case,'identity':case,'initial_test':metric,'curve':curve})
    return {'measurements':rows,'smoke_only':False,'epochs':5,'recovery_images':45000,'backend':backend}

@pytest.mark.parametrize('backend',['pcm','om'])
def test_full_matrix_required(backend):
    value=result(backend);assert len(validate(value))==270
    value['measurements'].pop()
    with pytest.raises(ValueError,match='Missing/duplicate'):validate(value)

@pytest.mark.parametrize('change',['pairing','writes','examples','epochs','identities'])
def test_mismatched_comparisons_rejected(change):
    value=deepcopy(result());row=value['measurements'][1]
    if change=='pairing':row['p0_sha256']='different'
    elif change=='writes':row['curve'][0]['cost']['array_reprogram_calls']=1
    elif change=='examples':row['curve'][0]['test']['examples']=5000
    elif change=='epochs':row['curve'].pop()
    else:
        for r in value['measurements']:
            if r['source']=='standard_hwa' and r['case']=='nominal':r['identity']='other devices'
    with pytest.raises(ValueError):validate(value)


def test_paired_effect_sign_and_array_count():
    rows=[{'case':'high','array_seed':i,'kl_improvement':x,'accuracy_gain_pp':1.} for i,x in enumerate((.1,.2,-.03))]
    r=aggregate(rows,['case'],['kl_improvement','accuracy_gain_pp'])[0]
    assert r['kl_improvement_positive_arrays']==2
    assert r['accuracy_gain_pp_positive_arrays']==3
    assert r['kl_improvement_mean']==pytest.approx(.09)
    with pytest.raises(ValueError):aggregate(rows[:2],['case'],['kl_improvement'])


@pytest.mark.parametrize('invalid',[float('nan'),float('inf')])
def test_nonfinite_evidence_cannot_enter_report(invalid):
    value=deepcopy(result())
    value['measurements'][1]['curve'][0]['test']['teacher_kl']=invalid
    with pytest.raises(ValueError,match='Nonfinite'):validate(value)
    rows=[{'case':'high','array_seed':i,'kl_improvement':x} for i,x in enumerate((.1,invalid,.2))]
    with pytest.raises(ValueError,match='Nonfinite'):aggregate(rows,['case'],['kl_improvement'])
