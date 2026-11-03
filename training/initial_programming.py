"""Initial-target and observable pair-selection rules; no network runner."""
import numpy as np
import torch
from training.flowers_drift import interp, named_seed


def time_weights(times):
    times = np.asarray(times, float)
    widths = np.diff(times)
    weights = np.zeros(len(times))
    weights[:-1] += widths/2
    weights[1:] += widths/2
    return weights/(times[-1]-times[0])


def prebias_table(law, settings, anchor_time, *, points=8193, quadrature=4096):
    """Global grid minimization of integrated squared *mean* pair error."""
    g = np.linspace(9, 88.2, points)
    q = np.linspace(-1, 1, points)
    ages = settings.release_seconds+np.asarray(settings.evaluation_seconds)-anchor_time
    means = law.expected_clipped(np.r_[g, 48.6], ages, quadrature)
    b = means[:-1]-means[-1]
    weights = time_weights(settings.evaluation_seconds)
    linear, constant = b@weights, (b*b)@weights
    targets = np.empty_like(q)
    for i in range(0, len(q), 128):
        delta = 39.6*q[i:i+128]
        objective = constant[:, None]-2*linear[:, None]*delta
        idx = objective.argmin(0)
        # Exact equal minima: prefer the original clean target, then lower index.
        for j, selected in enumerate(idx):
            ties = np.flatnonzero(objective[:, j] == objective[selected, j])
            if len(ties) > 1:
                idx[j] = ties[np.argmin(np.abs(g[ties]-(48.6+delta[j])))]
        targets[i:i+len(delta)] = g[idx]
    return q, targets


def targets(q, policy, table=None):
    ref = torch.full_like(q, 48.6)
    if policy == 'P6':
        delta = 39.6*q
        return torch.stack((88.2+delta.clamp_max(0), 88.2-delta.clamp_min(0)))
    active = 48.6+39.6*q
    if policy in ('P2', 'P4'):
        if table is None:
            raise ValueError('Prebias policy needs its declared age-dependent lookup.')
        active = interp(q, *table)
    return torch.stack((active, ref))


def select_pairs(observed, clean_q, scales, shape, fraction, *, random_seed=None):
    """Uses only apparent reads and frozen targets/scales, never latent state."""
    count = int(fraction*clean_q.numel())
    if random_seed is None:
        error = (((observed[0]-observed[1])/39.6-clean_q).reshape(shape)*scales.reshape(-1, 1)).abs().flatten()
        indices = torch.argsort(error, descending=True, stable=True)[:count]
    else:
        generator = torch.Generator(device=clean_q.device).manual_seed(random_seed)
        indices = torch.randperm(clean_q.numel(), generator=generator, device=clean_q.device)[:count]
    mask = torch.zeros_like(clean_q, dtype=torch.bool)
    mask[indices] = True
    return mask


def observation_report(law, state, raw):
    source = law.to_source(state.anchor)
    bins = (torch.bucketize(source.flatten(), law.edges, right=True)-1).clamp(0, len(law.edges)-2)
    n = len(law.edges)-1
    delta = (raw-state.anchor).flatten().double()
    counts = torch.bincount(bins, minlength=n)
    sums = torch.bincount(bins, weights=delta, minlength=n)
    squares = torch.bincount(bins, weights=delta*delta, minlength=n)
    lower = torch.bincount(bins, weights=(raw.flatten() < 9).double(), minlength=n)
    upper = torch.bincount(bins, weights=(raw.flatten() > 88.2).double(), minlength=n)
    return torch.stack((counts.double(), sums, squares, lower, upper)).cpu().numpy()


def finish_observation_report(values, edges):
    rows = []
    for i in range(values.shape[1]):
        n, total, squares, low, high = values[:, i]
        rows.append({'source_g0_bounds_us': [edges[i], edges[i+1]], 'devices': int(n),
                     'raw_increment_mean_us': None if n == 0 else total/n,
                     'raw_increment_sd_us': None if n == 0 else max(0., squares/n-(total/n)**2)**.5,
                     'clipped_low': int(low), 'clipped_high': int(high),
                     'clipped_low_fraction': None if n == 0 else low/n,
                     'clipped_high_fraction': None if n == 0 else high/n})
    return rows
