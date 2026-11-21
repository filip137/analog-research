"""Task adaptation of native OPT DRN deployments, using observable calibration and fixed physical masks."""
import copy
import math

import torch

from hardware.devices.cmo import G_MAX_US, HALF_RANGE_US, MID_US
from hardware.devices.cmo_array import _load_flags, _pack_flags
from methods import fault_tuning as ft
from methods.programming.cmo import partner, rewrite_plan
from methods.task_adaptation import backward_batch


@torch.no_grad()
def output_calibration(model, view, tokens, device, microbatch, ideal_blocks=None):
    """Fit y_ideal = a*y_observed+b per channel, using precisely the same input for each pair."""
    calibration = ft.OutputCalibration(model).requires_grad_(False)
    hidden = [model.analog_input(chunk.to(device)) for chunk in tokens.split(microbatch)]
    for k, block in enumerate(model.blocks):
        reference = block if ideal_blocks is None else copy.deepcopy(block).requires_grad_(False)
        if ideal_blocks is not None:
            reference.load_state_dict(ideal_blocks[k])
        layer = model.digital.layers[model.start + k]
        observations = []
        moments = torch.zeros((4, block.bias2.numel()), dtype=torch.float64, device=device)
        count = 0
        for h in hidden:
            residual = layer.attention(h)
            x = layer.final_layer_norm(residual)
            ideal = reference(x)
            observed = block(x, view[k] if view is not None else None)
            a, b = observed.reshape(-1, observed.shape[-1]).double(), ideal.reshape(-1, ideal.shape[-1]).double()
            moments += torch.stack((a.sum(0), b.sum(0), a.square().sum(0), (a*b).sum(0)))
            count += len(a)
            observations.append((residual, observed))
        avg, target, second, product = moments / count
        variance = second - avg.square()
        gain = torch.where(variance > 1e-12, (product - avg*target) / variance.clamp_min(1e-12), 1.)
        calibration.gain[k].copy_(gain)
        calibration.shift[k].copy_(target - gain*avg)
        hidden = [res + y*calibration.gain[k] + calibration.shift[k] for res,y in observations]
    return calibration


def proxy_view(model, bank):
    """A static characterization on an independent development chip follows the moving digital targets.

    Detach physical values here: DrnMlp.conductances supplies the native straight-through gradient.
    Verify-flagged cells remain at their measured values throughout this empirical proxy fit.
    """
    views = []
    for u, array in zip(model.parameter_groups()[0], bank.arrays):
        target_us = MID_US + HALF_RANGE_US*u.detach()
        flagged = array.evidence['verify_flags'].to(u.device)
        observed = torch.where(flagged, array.held, (target_us + (array.held-array.target)).clamp_min(0.))
        views.append(observed / G_MAX_US)
    return tuple(zip(views[::2], views[1::2]))


def rank_masks(model, bank, forward, train, settings, progress):
    """One deterministic score pass, then nested global 1% and 10% masks over the same observed pool."""
    model.train().requires_grad_(False)
    masters = model.parameter_groups()[0]
    for u in masters:
        u.requires_grad_(True)
    scores = [torch.zeros_like(u) for u in masters]
    indices = torch.arange(settings.fisher_items)
    groups = indices.split(8)
    device = masters[0].device
    for step, group in enumerate(groups, 1):
        model.zero_grad(set_to_none=True)
        backward_batch(forward, train, group, device, settings.microbatch)
        for acc, u in zip(scores, masters):
            acc.addcmul_(u.grad, u.grad, value=1/len(groups))
        progress(step, len(groups))
    model.eval().requires_grad_(False).zero_grad(set_to_none=True)
    eligible = None if bank is None else [~a.evidence['verify_flags'].to(device) for a in bank.arrays]
    population = sum(u.numel() for u in masters) if eligible is None else sum(int(v.sum()) for v in eligible)
    fractions = {}
    smaller = None
    for fraction in (.01, .1):
        masks = ft.global_child_masks(scores, fraction, eligible=eligible)
        count = sum(int(mask.sum()) for mask in masks)
        if count != max(1, round(population*fraction)):
            raise RuntimeError('Global selection count mismatch.')
        if smaller is not None and any(bool((prior & ~new).any()) for prior,new in zip(smaller, masks)):
            raise RuntimeError('Global masks are not nested.')
        fractions[str(fraction)] = {'count': count, 'bits': [_pack_flags(mask) for mask in masks]}
        smaller = masks
    return {'eligible': population, 'fractions': fractions}


def unpack_masks(rank, fraction, model):
    masters = model.parameter_groups()[0]
    packed = rank['fractions'][str(fraction)]
    if len(packed['bits']) != len(masters):
        raise ValueError('Number of ranked arrays differs from model.')
    masks = [_load_flags({'mask_bits': bits}, 'mask', u.shape).to(u.device)
             for bits, u in zip(packed['bits'], masters)]
    if sum(int(m.sum()) for m in masks) != packed['count']:
        raise ValueError('Restored global mask count mismatch.')
    return masks


def lr_factor(step, steps):
    """Six-percent linear warm-up, then cosine decay; step is zero based."""
    warmup = max(1, round(.06*steps))
    if step < warmup:
        return (step+1)/warmup
    return .5*(1+math.cos(math.pi*(step-warmup+1)/max(1,steps-warmup)))


def train(model, bank, view_fn, bundle, s, masks, physical, progress):
    """Return the still-attached LoRA and physical update counters when applicable."""
    model.train().requires_grad_(False)
    masters = model.parameter_groups()[0]
    adapters, writer = None, None
    if s.method == 'lora':
        adapters = ft.Adapters(model, 16, s.seed)
        adapters.attach(model)
        params = list(adapters.parameters())
    elif s.method == 'child':
        if physical:
            writer = ft.PhysicalArm(model, bank, 'child_d', s.lr, s.steps, masks, 2., gate='fixed_mid')
        params = masters
    else:
        params = list(model.parameters())
    for p in params:
        p.requires_grad_(True)
    optimizer = None if writer else torch.optim.AdamW(params, lr=s.lr, weight_decay=0., foreach=False)
    grad_norm = None
    for step, indices in enumerate(bundle['order'][:s.steps]):
        view = view_fn()
        forward = lambda tokens: model(tokens, view)
        model.zero_grad(set_to_none=True)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        loss = backward_batch(forward, bundle['splits']['train'], indices.cpu(), params[0].device, s.microbatch)
        if masks is not None:
            for u, mask in zip(masters, masks):
                u.grad.masked_fill_(~mask, 0.)
        if writer:
            for w, u in zip(writer.writers, masters):
                w.step(u.grad)
        else:
            maximum = float('inf') if s.method == 'child' else 1.
            grad_norm = torch.nn.utils.clip_grad_norm_(params, maximum, error_if_nonfinite=True)
            factor = .5*(1+math.cos(math.pi*step/s.steps)) if s.method == 'child' else lr_factor(step, s.steps)
            for group in optimizer.param_groups:
                group['lr'] = s.lr*factor
            optimizer.step()
            if s.method != 'lora':
                with torch.no_grad():
                    for i, u in enumerate(masters):
                        if masks is None:
                            u.clamp_(-1, 1)
                        else:
                            u[masks[i]] = u[masks[i]].clamp(-1, 1)
        progress(step+1, s.steps, loss)
    if len(bundle['order']) < s.steps:
        raise ValueError('Insufficient updates in the pinned order.')
    model.zero_grad(set_to_none=True)
    if writer:
        report = writer.write_report()
        report['attempts_outside_mask'] = sum(int(w.attempt_counts[~m].sum()) for w,m in zip(writer.writers,masks))
        report['distinct_written'] = sum(int(((w.attempt_counts > 0) & ~a._mask).sum())
                                         for w,a in zip(writer.writers, bank.arrays))
        if report['attempts_outside_mask']:
            raise RuntimeError('Child-Tuning wrote outside the selected physical mask.')
        report['optimizer_bytes'] = sum(getattr(w,key).numel()*getattr(w,key).element_size()
                                        for w in writer.writers for key in w.state_names)
        report['selected'] = sum(int(m.sum()) for m in masks)
        report['verify_reads'] = 0
        report['transferred_bytes'] = 0
        writer.close()
    else:
        report = {'gradient_norm_last': float(grad_norm),
                  'optimizer_bytes': sum(t.numel()*t.element_size() for state in optimizer.state.values()
                                         for t in state.values() if isinstance(t,torch.Tensor)),
                  'selected':sum(int(m.sum()) for m in masks) if masks is not None else None}
    model.eval().requires_grad_(False)
    return adapters, report


@torch.no_grad()
def export_fit(model, method, adapters, masks):
    if method == 'lora':
        return {'adapter': {k:v.detach().cpu() for k,v in adapters.state_dict().items()}}
    masters = model.parameter_groups()[0]
    if method == 'child':
        return {'cells': [{'indices': torch.nonzero(m.flatten()).flatten().cpu(), 'targets': u[m].cpu()}
                          for u,m in zip(masters,masks)]}
    return {'model': {k:v.detach().cpu() for k,v in model.state_dict().items()}}


@torch.no_grad()
def transfer(model, bank, fit, method, seed):
    """Apply sparse/full absolute targets to the original chip and enforce mask on verify compensation."""
    if method == 'lora':
        adapters = ft.Adapters(model, 16, seed)
        adapters.load_state_dict(fit['adapter'])
        adapters.attach(model)
        return adapters, {'costs': bank.costs(), 'transferred_bytes': sum(p.numel()*p.element_size() for p in adapters.parameters())}
    masters = model.parameter_groups()[0]
    updates = []
    if method == 'full':
        # Only deploy changed targets. The trained state also carries all changed digital tensors.
        master_names = {id(p):name for name,p in model.named_parameters()}
        for u in masters:
            trained = fit['model'][master_names[id(u)]].to(u)
            updates.append((trained, torch.ones_like(u, dtype=torch.bool)))
        model.load_state_dict(fit['model'])
    else:
        if len(fit['cells']) != len(masters):
            raise ValueError('Incorrect number of sparse target arrays.')
        for u, entry in zip(masters, fit['cells']):
            ids, values = entry['indices'].to(u.device), entry['targets'].to(u)
            if ids.ndim != 1 or len(ids) != len(values) or len(ids) != len(ids.unique()) or bool((ids < 0).any() or (ids >= u.numel()).any()):
                raise ValueError('Invalid sparse coordinate list.')
            trained = u.detach().clone()
            trained.flatten()[ids] = values
            permitted = torch.zeros_like(u, dtype=torch.bool)
            permitted.flatten()[ids] = True
            updates.append((trained, permitted))
    touched = max_wear = reads = outside = 0
    for (trained, permitted), array in zip(updates, bank.arrays):
        targets = MID_US + HALF_RANGE_US*trained
        selected = permitted & (targets != array.target)
        if not selected.any():
            continue
        array._program(targets, selected)
        # Use the native partner axis (arrays are stored transposed relative to the P&V controller).
        second, requested, _, _ = rewrite_plan(array.held.T, targets.T, noise_scale=array.noise_scale)
        pairs = selected | partner(selected.T).T
        second = second.T & permitted & pairs
        reads += int(pairs.sum())
        outside += int(((selected | second) & ~permitted).sum())
        array._program(requested.T, second)
        writable = ~array._mask  # cost audit only
        touched += int(((selected | second) & writable).sum())
        max_wear = max(max_wear, int((selected.to(torch.int8) + second).max()))
        array.check()
    bytes_sent = sum(t.numel()*t.element_size() for t in fit['model'].values()) if method == 'full' else sum(
        v.numel()*v.element_size() for entry in fit['cells'] for v in entry.values())
    return None, {'costs': bank.costs(), 'distinct_written': touched, 'max_writes_per_cell': max_wear,
                  'verify_reads': reads, 'attempts_outside_mask': outside, 'transferred_bytes': bytes_sent}
