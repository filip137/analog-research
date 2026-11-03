"""methods.maintenance.selection — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch


def _smallest(score, eligible, count):
    ids = torch.nonzero(eligible.flatten(), as_tuple=False).flatten()
    if len(ids) < count:
        raise ValueError(f'Only {len(ids)} uncensored cells for a requested mask of {count}.')
    # Stable ties preserve physical index order (including the no-drift control).
    return ids[torch.argsort(score.flatten()[ids], stable=True)[:count]]


def select_masks(cells, groups, roles, fraction, seed):
    """10% of *all* cells per MLP, with observable-pool matched controls.

    groups and roles are physical-cell tensors with integer labels supplied by
    the architecture. Random and oracle masks match the estimated mask's counts
    in every MLP x rail-role x initial-conductance-decile stratum.
    """
    scores, eligible = zip(*(c.observed_stability() for c in cells))
    sizes = [c.target.numel() for c in cells]
    device = cells[0].target.device
    score = torch.cat([v.flatten() for v in scores])
    pool = torch.cat([v.flatten() for v in eligible])
    group = torch.cat([v.flatten().to(device) for v in groups])
    role = torch.cat([v.flatten().to(device) for v in roles])
    decile = torch.cat([((c.target.flatten()-c.law.g_min)/(c.law.g_max-c.law.g_min)*10)
                        .long().clamp(0, 9) for c in cells])
    z = torch.cat([c.persistent.flatten().abs() for c in cells])
    chosen = torch.zeros_like(pool)
    for g in group.unique():
        in_group = group == g
        k = max(1, int(int(in_group.sum()) * fraction))
        chosen[_smallest(score, pool & in_group, k)] = True
    random_mask, oracle = torch.zeros_like(pool), torch.zeros_like(pool)
    strata = group.to(torch.int64)*1000 + role.to(torch.int64)*10 + decile
    rng = torch.Generator(device=device).manual_seed(seed)
    for stratum in strata[chosen].unique(sorted=True):
        available = pool & (strata == stratum)
        k = int((chosen & (strata == stratum)).sum())
        ids = torch.nonzero(available, as_tuple=False).flatten()
        random_mask[ids[torch.randperm(len(ids), generator=rng, device=device)[:k]]] = True
        oracle[_smallest(z, available, k)] = True
    def split_indices(mask):
        return [torch.nonzero(v, as_tuple=False).flatten() for v in mask.split(sizes)]
    return {'stable': split_indices(chosen), 'random': split_indices(random_mask),
            'oracle': split_indices(oracle)}, {
                'physical_devices': sum(sizes), 'selected_devices': int(chosen.sum()),
                'eligible_devices': int(pool.sum()), 'excluded_devices': int((~pool).sum()),
                'selection_reads': sum(sizes), 'matched_strata': 'MLP/physical-role/target-decile'}
