"""Analysis-only numerical comparisons; no experiment execution or selection."""
import torch
from torch.nn import functional as F


def compare_logits(actual, expected, teacher_probabilities):
    """Check FP32 agreement, including cancellation near zero among large logits.

    Preserve the original elementwise check in the receipt. Its exceptions must
    satisfy a row-scale FP32 roundoff bound, global relative error, identical
    predictions and agreement of the actual scientific KL metric.
    """
    if actual.shape != expected.shape or actual.dtype != torch.float32 or expected.dtype != torch.float32:
        raise ValueError('Expected same-shape FP32 logits.')
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('Nonfinite logits.')
    difference=(actual-expected).abs()
    ordinary=difference <= 2e-5+2e-4*expected.abs()
    row_scale=torch.maximum(actual.abs().amax(1,keepdim=True),expected.abs().amax(1,keepdim=True))
    roundoff=32*torch.finfo(torch.float32).eps*row_scale
    relative=float(difference.double().norm()/expected.double().norm().clamp_min(1e-30))
    changed=int((actual.argmax(1)!=expected.argmax(1)).sum())
    # Accumulate the diagnostic KL in FP64 to expose FP32 reduction cancellation.
    probabilities=teacher_probabilities.double()
    kl=lambda logits:float(F.kl_div(logits.double().log_softmax(1),probabilities,reduction='batchmean'))
    reference_kl=kl(expected);actual_kl=kl(actual)
    result={'maximum_logit_error':float(difference.max()),'relative_l2_error':relative,
            'prediction_changes':changed,'original_elementwise_exceptions':int((~ordinary).sum()),
            'teacher_kl_reference':reference_kl,'teacher_kl_actual':actual_kl,
            'teacher_kl_absolute_error':abs(actual_kl-reference_kl),
            'rule':'elementwise rtol2e-4/atol2e-5 OR 32*FP32eps*row_scale; relativeL2<=1e-6; predictions identical; KL atol2e-5/rtol1e-6'}
    if not bool((ordinary | (difference<=roundoff)).all()) or relative>1e-6 or changed or abs(actual_kl-reference_kl)>2e-5+1e-6*abs(reference_kl):
        raise ValueError('GPU numerical agreement failed: '+str(result))
    return result
