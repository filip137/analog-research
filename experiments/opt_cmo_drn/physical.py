"""Serializable held CMO DRN arrays; raw programming and recovery stay distinct."""

import copy
import math
import numpy as np
import torch

from experiments.opt_cmo_drn.faults import tensor_hash
from model.resistive.differentiable import coefficients
from training.reram_cmo import G_MIN_US, G_MAX_US, MID_US, HALF_RANGE_US, programming_sigma


def _pack_flags(value):
    return torch.from_numpy(np.packbits(value.detach().cpu().numpy().reshape(-1),bitorder='little'))


def _load_flags(state,name,shape):
    if name in state:
        value=state[name].cpu()
        if value.dtype!=torch.bool or tuple(value.shape)!=tuple(shape):
            raise ValueError(f'Invalid saved physical flag: {name}.')
        return value.clone()
    bits=state[name+'_bits']
    count=math.prod(shape)
    if bits.dtype!=torch.uint8 or bits.ndim!=1 or bits.numel()!=(count+7)//8:
        raise ValueError(f'Invalid packed physical flag: {name}.')
    return torch.from_numpy(np.unpackbits(bits.cpu().numpy(),count=count,bitorder='little').reshape(shape).astype(bool))


class ConductanceArray:
    def __init__(self,state,device):
        self.name=state['name']
        self.noise_scale=state['noise_scale']
        self.held=state['held'].to(device).clone()
        self.target=state['target'].to(device).clone()
        self._mask=_load_flags(state,'mask',self.held.shape).to(device)
        if 'stuck' in state:
            if 'stuck_values' in state:raise ValueError('Ambiguous saved fault representation.')
            self._stuck=state['stuck'].to(device).clone()
        else:
            values=state['stuck_values']
            if values.ndim!=1 or values.numel()!=int(self._mask.sum()) or values.dtype!=self.held.dtype:
                raise ValueError('Saved corrupt-cell values do not match the fault mask.')
            self._stuck=torch.zeros_like(self.held)
            self._stuck[self._mask]=values.to(device)
        # Original verify observations and raw reference are logging evidence.
        self.evidence={'raw':state['raw'].cpu().clone(),
                       **{k:_load_flags(state,k,self.held.shape) for k in ('verify_flags','rewrite_selected')}}
        self.generator=torch.Generator(device=device)
        self.generator.set_state(state['rng'].cpu())
        self.counters={k:int(state[k]) for k in ('write_attempts','device_writes','stuck_write_attempts')}
        self.counters['gmin_clipped_writes']=int(state.get('gmin_clipped_writes',0))
        self.bounded_counters=copy.deepcopy(state.get('bounded_counters',{}))
        self.check()

    def read(self):
        return (self.held-MID_US)/HALF_RANGE_US

    def check(self):
        if not bool(torch.isfinite(self.held).all() and (self.held>=0).all()):
            finite=torch.isfinite(self.held)
            minimum=float(self.held[finite].min()) if bool(finite.any()) else None
            raise FloatingPointError(f'Invalid physical conductances in {self.name}: '
                f'negative={int((self.held<0).sum())}, nonfinite={int((~finite).sum())}, '
                f'minimum_finite_us={minimum}; no clipping or resampling.')
        if not torch.equal(self.held[self._mask],self._stuck[self._mask]):
            raise ValueError('Permanent fault state changed.')

    @torch.no_grad()
    def _program(self,requested,selected,*,noise_factor=1.,conductance_floor='none'):
        if conductance_floor not in ('none','gmin'):raise ValueError('Unknown conductance floor.')
        if requested.shape!=self.held.shape or selected.shape!=self.held.shape:
            raise ValueError('Physical command shape mismatch.')
        if not bool(torch.isfinite(requested).all()):
            raise FloatingPointError('Invalid physical target; no clipping or resampling.')
        noise=torch.randn(requested.shape,device=requested.device,generator=self.generator)
        # Draw even in the noiseless control to preserve the matched RNG stream.
        # Keep the original expression bit-exact when noise is enabled.
        drawn=requested if noise_factor==0. else requested+self.noise_scale*programming_sigma(requested,2.0)*noise
        written=selected&~self._mask
        if not bool(torch.isfinite(drawn[written]).all()):
            raise FloatingPointError(f'Nonfinite post-write conductance in {self.name}; clipping cannot repair it.')
        if conductance_floor=='gmin':
            self.counters['gmin_clipped_writes']+=int((written&(drawn<G_MIN_US)).sum())
            drawn=drawn.clamp_min(G_MIN_US)
        self.held=torch.where(written,drawn,self.held)
        self.counters['write_attempts']+=int(selected.sum())
        self.counters['device_writes']+=int(written.sum())
        self.counters['stuck_write_attempts']+=int((selected&self._mask).sum())
        self.check()

    def write(self,command,*,noise_factor=1.,conductance_floor='none',target_bounds='none'):
        if noise_factor not in (0.,1.):raise ValueError('Expected recovery write noise factor 0/1.')
        if target_bounds not in ('none','gmin_gmax'):raise ValueError('Unknown target bounds.')
        requested=self.held+HALF_RANGE_US*command
        selected=command!=0
        if target_bounds=='none':
            self._program(requested,selected,noise_factor=noise_factor,conductance_floor=conductance_floor)
            return
        if conductance_floor!='none':raise ValueError('Bounded targets cannot use a realized-state floor.')
        if command.shape!=self.held.shape or not bool(torch.isfinite(requested).all()):
            raise FloatingPointError('Invalid intended bounded write target.')
        bounded=requested.clamp(G_MIN_US,G_MAX_US)
        attempted=selected&(bounded!=self.held)
        counts={'proposed_commands':int(selected.sum()),
                'target_clipped_low':int((selected&(requested<G_MIN_US)).sum()),
                'target_clipped_high':int((selected&(requested>G_MAX_US)).sum()),
                'projected_noops':int((selected&~attempted).sum())}
        before=self.held
        # A full draw advances identically even when the selected set is empty.
        self._program(bounded,attempted,noise_factor=noise_factor)
        counts['state_changing_writes']=int((self.held!=before).sum())
        for key,value in counts.items():self.bounded_counters[key]=self.bounded_counters.get(key,0)+value
        return attempted

    def statistics(self):
        return {'name':self.name,'minimum_us':float(self.held.min()),'maximum_us':float(self.held.max()),
                'mean_us':float(self.held.mean()),'std_us':float(self.held.std(unbiased=False)),
                'below_gmin':int((self.held<G_MIN_US).sum()),'above_gmax':int((self.held>G_MAX_US).sum()),
                **self.bounded_counters}

    def rewrite(self,*,conductance_floor='none'):
        self._program(self.target,torch.ones_like(self._mask),conductance_floor=conductance_floor)

    def state_dict(self):
        return {'name':self.name,'noise_scale':self.noise_scale,'rng':self.generator.get_state().cpu(),
                'stuck_values':self._stuck[self._mask].detach().cpu().clone(),
                'mask_bits':_pack_flags(self._mask),
                **{k:v.detach().cpu().clone() for k,v in
                   (('held',self.held),('target',self.target),('raw',self.evidence['raw']))},
                **{k+'_bits':_pack_flags(self.evidence[k]) for k in ('verify_flags','rewrite_selected')},**self.counters,
                **({'bounded_counters':copy.deepcopy(self.bounded_counters)} if self.bounded_counters else {})}


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
