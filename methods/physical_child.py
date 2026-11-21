"""Global sparse fine-tuning in individual physical conductance coordinates on a two-sided crossbar."""
from __future__ import annotations

import math
import torch

from core.seeds import binding_seed
from core.tensor_identity import tensor_hash
from hardware.devices import cmo
from methods.programming.lifecycle import detect_stuck_devices, pair_rewrite_plan


def nested_masks(logical_fisher, eligible, layout, fractions, span, tile_size=256):
    """Two-sided dL/dG legs have opposite signs and identical squared gradients."""
    if eligible.shape != (2, logical_fisher.numel()) or eligible.dtype != torch.bool:
        raise ValueError("Invalid controller eligibility map.")
    parts = []
    for matrix in layout:
        s = slice(matrix.offset, matrix.offset + matrix.size)
        score = (logical_fisher[s] / span ** 2).expand(2, -1)
        parts.append(score.masked_fill(~eligible[:, s], -torch.inf).reshape(-1))
    scores = torch.cat(parts)
    if not torch.isfinite(logical_fisher).all() or not eligible.any():
        raise ValueError("Invalid Fisher scores or empty eligible population.")
    total = int(eligible.sum())
    masks = {}
    for fraction in fractions:
        k = max(1, round(fraction * total))
        cutoff = scores.topk(k, sorted=False).values.min()
        chosen = scores > cutoff
        ties = torch.nonzero(scores == cutoff).flatten()
        need = k - int(chosen.sum())
        if 0 < need < len(ties):
            # Native physical_arrays order is matrix, output tile, input tile, split-half,
            # then role and flattened weight. Only exact cutoff ties need sorting.
            sorted_ties = []
            offset = 0
            for m in layout:
                local = ties[(ties >= offset) & (ties < offset + 2 * m.size)] - offset
                if local.numel():
                    cols = m.shape[1]
                    weight = local % m.size
                    row, col = weight // cols, weight % cols
                    tile = (row // tile_size) * math.ceil(cols / tile_size) + col // tile_size
                    width = (cols - (col // tile_size) * tile_size).clamp_max(tile_size)
                    part = (col % tile_size >= (width + 1) // 2).long()
                    keys = (tile * 2 + part) * (2 * m.size) + local
                    sorted_ties.append(local[keys.argsort()] + offset)
                offset += 2 * m.size
            ties = torch.cat(sorted_ties)
        chosen[ties[:need]] = True
        per_matrix = chosen.split([m.size * 2 for m in layout])
        mask = torch.cat([m.reshape(2, -1) for m in per_matrix], 1)
        if int(mask.sum()) != k or (mask & ~eligible).any():
            raise RuntimeError("Physical Fisher selection failed.")
        masks[fraction] = mask
    return masks


def held_conductances(array):
    """Observable values, assembled for read simulation; this operation does not perform a verify scan."""
    return torch.cat([child.g for child in array.children], 1)


class PhysicalWriter:
    """Programs both physical legs independently, with a command-side mask audit."""
    def __init__(self, array, mask, namespace, seed):
        self.array, self.mask = array, mask
        self.namespace, self.seed = namespace, seed
        self.attempts = 0
        self.rejects = 0
        self.verify_reads = 0
        self.counts = torch.zeros_like(mask, dtype=torch.int32)
        self.event = 0

    @torch.no_grad()
    def program(self, targets, requested):
        if requested.shape != self.mask.shape or bool((requested & ~self.mask).any()):
            raise RuntimeError("Requested a physical write outside the frozen device mask.")
        selected_n = int(requested.sum())
        if not selected_n:
            return
        seeds = {m.name: binding_seed(self.seed, self.namespace, f"task-write/{self.event}/{m.name}")
                 for m in self.array.layout}
        bounded = targets.clamp(cmo.G_MIN_US, cmo.G_MAX_US)
        self.array.program_devices(bounded, requested, seeds)
        self.event += 1
        self.attempts += selected_n
        # Hidden fault state is used strictly after the command, to audit incurred cost.
        for m, child in zip(self.array.layout, self.array.children):
            s = slice(m.offset, m.offset + m.size)
            selected = requested[:, s]
            self.counts[:, s] += selected & ~child.mask
            self.rejects += int((selected & child.mask).sum())

    def upload(self, target, desired_q, expected, flags, span):
        """Program the transmitted targets; verify and compensate a partner only inside the support."""
        affected = self.mask & (target != expected)
        self.program(target, affected)
        observed = self.array.read_devices()
        self.verify_reads += observed.numel()
        fresh = detect_stuck_devices(observed, desired_q, acceptance=self.array.children[0].acceptance,
                                     noise_scale=self.array.children[0].noise_scale, threshold_sigma=5.,
                                     expected=target)
        combined = torch.where(affected, fresh, flags)
        selected, targets, _ = pair_rewrite_plan(combined, observed, desired_q, "pair", span)
        self.program(targets, self.mask & selected & affected.any(0))

    def cost(self):
        self.array.verify_permanence()
        successes = int(self.counts.sum())
        if self.attempts != successes + self.rejects or (self.counts[~self.mask] != 0).any():
            raise RuntimeError("Physical command and realized write audits disagree.")
        return {"attempted_device_writes": self.attempts, "successful_device_writes": successes,
                "stuck_write_attempts": self.rejects, "distinct_devices_written": int((self.counts > 0).sum()),
                "max_writes_per_device": int(self.counts.max()), "verify_reads": self.verify_reads,
                "out_of_mask_attempts": 0, "mask_sha256": tensor_hash(self.mask),
                "selected_devices": int(self.mask.sum())}
