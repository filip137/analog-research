"""Readout calibration at fixed DRN mapping and training cohort."""
import torch

@torch.no_grad()
def fit_mapping(model, source, tokens, max_tokens=256, progress=None):
    """Fit readout gains at fixed full conductance span on the training cohort."""
    model.eval()
    source.eval()
    hidden=source.embed(tokens)
    report=[]
    for i,layer in enumerate(source.layers):
        hidden=layer.attention(hidden)
        x=layer.final_layer_norm(hidden)
        target=source.branch(i,x)
        if i>=source.start:
            block=model.blocks[i-source.start]
            sample=x.reshape(-1,x.shape[-1])[:max_tokens]
            y=target.reshape(-1,target.shape[-1])[:max_tokens]-block.bias2
            block.initialize(layer.fc1.weight,layer.fc2.weight)
            prediction=block(sample)-block.bias2
            gain=float(((prediction*y).sum()/prediction.square().sum().clamp_min(1e-30)).clamp_min(1e-12))
            loss=float((prediction*gain-y).square().sum()/y.square().sum().clamp_min(1e-30))
            if progress: progress({'layer':i,'fractions':(1.0,1.0),'relative_squared_error':loss})
            block.readout_scale.fill_(gain)
            current=torch.cat((block.bias1,-block.bias1))*block.current_scale
            diagnostics=block.solver.diagnostics(sample,*block.conductances(),current,block.gains[0])
            report.append({'layer':i,'fractions':(1.0,1.0),'relative_squared_error':loss,'readout_scale':gain,
                           'diagnostics':diagnostics})
        hidden=hidden+target
    return report
