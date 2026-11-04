"""Serializable held CMO DRN arrays; raw programming and recovery stay distinct."""

import copy
import math
import numpy as np
import torch

from hardware.faults import tensor_hash
from networks.resistive.differentiable import coefficients
from hardware.devices.cmo import G_MIN_US, G_MAX_US, MID_US, HALF_RANGE_US, programming_sigma


from hardware.devices.cmo_array import _pack_flags


from hardware.devices.cmo_array import _load_flags


from hardware.devices.cmo_array import ConductanceArray


class CmoDrnBank:
    def __init__(self,state,device):
        if state.get('schema') not in ('opt_cmo_drn.bank.v1','opt_cmo_drn.bank.v2'):
            raise ValueError('Expected a saved native DRN deployment.')
        self.receipt=copy.deepcopy(state['receipt'])
        self.arrays=[ConductanceArray(s,device) for s in state['cells']]

    def view(self,model,*,raw=False):
        values=[(a.evidence['raw'].to(a.held.device) if raw else a.held)/G_MAX_US for a in self.arrays]
        if len(values)!=2*len(model.blocks):
            raise ValueError('Physical array count differs from the DRN model.')
        pairs=tuple(tuple(values[i:i+2]) for i in range(0,len(values),2))
        for block,(g1,g2) in zip(model.blocks,pairs):
            if not all(bool(torch.isfinite(g).all() and (g>=0).all()) for g in (g1,g2)):
                raise FloatingPointError('Invalid held physical DRN view.')
            if g1.shape!=block.u1.shape or g2.shape!=block.u2.shape:
                raise ValueError('Physical array shape differs from the DRN model.')
            dh,do=coefficients(g1,g2,model.settings)
            if not bool(torch.isfinite(dh).all() and torch.isfinite(do).all() and (dh>0).all() and (do>0).all()):
                raise FloatingPointError('Nonpositive or nonfinite physical DRN denominator.')
        return pairs

    def held_hashes(self):
        return [tensor_hash(a.held) for a in self.arrays]

    def costs(self):
        counts={key:sum(a.counters[key] for a in self.arrays) for key in
                ('write_attempts','device_writes','stuck_write_attempts','gmin_clipped_writes')}
        for key in {k for a in self.arrays for k in a.bounded_counters}:
            counts[key]=sum(a.bounded_counters.get(key,0) for a in self.arrays)
        devices=sum(a.held.numel() for a in self.arrays)
        return {**counts,'physical_devices':devices,'logical_weights':devices//4,
                'attempts_per_device':counts['write_attempts']/devices,
                'attempts_per_logical_weight':4*counts['write_attempts']/devices,
                'initial_programming_attempts':devices,
                'initial_partner_rewrite_attempts':sum(r['rewrite_attempts'] for r in self.receipt['matrices'])}

    def state_dict(self):
        return {'schema':'opt_cmo_drn.bank.v2','receipt':copy.deepcopy(self.receipt),
                'cells':[a.state_dict() for a in self.arrays]}
