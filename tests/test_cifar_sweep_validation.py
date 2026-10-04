import pytest
import torch
from experiments.cifar_crossbar.sweep_validation import compare_logits


def test_cancellation_among_large_logits_is_recorded_not_hidden():
    expected=torch.tensor([[200000.,-10.,-100000.]])
    actual=expected.clone();actual[0,1]+=.003
    result=compare_logits(actual,expected,torch.full_like(expected,1/3))
    assert result['original_elementwise_exceptions']==1
    assert result['prediction_changes']==0
    assert result['relative_l2_error']<1e-7


@pytest.mark.parametrize('failure',['large_error','changed_prediction','nonfinite','kl'])
def test_material_gpu_disagreement_is_rejected(failure):
    expected=torch.tensor([[1.,1.,-1.]])
    actual=expected.clone();teacher=torch.full_like(expected,1/3)
    if failure=='large_error':actual[0,2]+=.01
    elif failure=='changed_prediction':actual[0,1]+=1e-7
    elif failure=='nonfinite':actual[0,0]=float('nan')
    else:
        expected=torch.tensor([[200000.,0.,199999.9]])
        actual=expected.clone();actual[0,2]-=.03125
        teacher=torch.tensor([[0.,0.,1.]])
    with pytest.raises(ValueError):compare_logits(actual,expected,teacher)
