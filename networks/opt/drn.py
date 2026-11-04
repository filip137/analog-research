"""Reconstruct the pinned whole-model CMO source and map its six MLPs."""

import copy
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from networks.opt.decoder import OptDecoder
from networks.resistive.differentiable import DifferentiableDrn, DrnSettings
from hardware.devices.cmo import G_MIN_US, G_MAX_US, MID_US, HALF_RANGE_US


class SourceDecoder(OptDecoder):
    """CMO digital master reconstructed in logical weight units; gains separate."""

    def branch(self, index, x):
        layer = self.layers[index]
        k = index - self.start
        if k < 0:
            return layer.fc2(F.relu(layer.fc1(x)))
        hidden = F.linear(x, layer.fc1.weight) * self.gains[2*k] + layer.fc1.bias
        return F.linear(F.relu(hidden), layer.fc2.weight) * self.gains[2*k+1] + layer.fc2.bias

    def forward(self, tokens):
        hidden = self.embed(tokens)
        for i, layer in enumerate(self.layers):
            hidden = layer.attention(hidden)
            hidden = hidden + self.branch(i, layer.final_layer_norm(hidden))
        return self.logits(hidden)


def reconstruct_source(teacher, payload, analog_layers=6):
    state = payload['model']
    if payload['arm']['scope'] != 'model':
        raise ValueError('Expected a whole-model CMO source, including digital weights.')
    source = SourceDecoder(teacher.config)
    source.load_state_dict(teacher.state_dict(), strict=True)
    source.start = len(source.layers) - analog_layers
    expected_digital = {k for k in teacher.state_dict() if not any(
        k.startswith(f'layers.{i}.{module}.') for i in range(source.start,len(source.layers))
        for module in ('fc1','fc2','final_layer_norm'))}
    actual_digital = {k.removeprefix('digital.'): v for k,v in state.items() if k.startswith('digital.')}
    if set(actual_digital) != expected_digital:
        raise ValueError('Source digital weights do not match the declared OPT architecture.')
    restored = source.state_dict()
    restored.update(actual_digital)
    offset = channel_offset = 0
    for k, i in enumerate(range(source.start, len(source.layers))):
        for j, name in enumerate(('fc1','fc2')):
            index = 2*k+j
            linear = getattr(source.layers[i],name)
            count = linear.weight.numel()
            q = state['q'][offset:offset+count].reshape_as(linear.weight)
            if 'channel_scales' in state:
                scale = state['channel_scales'][channel_offset:channel_offset+q.shape[0],None]
            else:
                scale = state['scales'][index]
            restored[f'layers.{i}.{name}.weight'] = q * scale
            restored[f'layers.{i}.{name}.bias'] = state[f'mlp_biases.{index}']
            offset += count
            channel_offset += q.shape[0]
        for name in ('weight','bias'):
            restored[f'layers.{i}.final_layer_norm.{name}'] = state[f'mlp_norms.{k}.{name}']
    if offset != state['q'].numel():
        raise ValueError('Source has extra or missing analog weights.')
    source.load_state_dict(restored,strict=True)
    source.register_buffer('gains',state['output_gains'].clone())
    return source.eval().requires_grad_(False)


def lift(weight, paired=False):
    """Same positive/negative lift as mnist_relu_drn.components, in uS."""
    w = weight.T
    p, n = w.clamp_min(0), (-w).clamp_min(0)
    if paired:
        return torch.cat((torch.stack((p,n),-1).flatten(1), torch.stack((n,p),-1).flatten(1)),0)
    return torch.cat((torch.cat((p,n),1),torch.cat((n,p),1)),0)


class DrnMlp(nn.Module):
    def __init__(self, fc1, fc2, gains, settings):
        super().__init__()
        self.solver = DifferentiableDrn(settings)
        self.u1 = nn.Parameter(fc1.weight.new_empty((2*fc1.in_features,2*fc1.out_features)))
        self.u2 = nn.Parameter(fc2.weight.new_empty((2*fc2.in_features,2*fc2.out_features)))
        self.bias1 = nn.Parameter(fc1.bias.detach().clone())
        self.bias2 = nn.Parameter(fc2.bias.detach().clone())
        self.gains = nn.Parameter(gains.detach().clone())
        self.register_buffer('current_scale',fc1.weight.new_tensor(1.0))
        self.register_buffer('readout_scale',fc1.weight.new_tensor(1.0))
        self.initialize(fc1.weight,fc2.weight)

    @torch.no_grad()
    def initialize(self, w1, w2):
        """Use the full target window independently for each logical matrix."""
        for u,w,paired in ((self.u1,w1,False),(self.u2,w2,True)):
            scale = (G_MAX_US-G_MIN_US) / w.abs().max().clamp_min(1e-12)
            target = G_MIN_US + scale * lift(w,paired)
            u.copy_((target-MID_US)/HALF_RANGE_US)
            if not paired:
                self.current_scale.copy_(scale/G_MAX_US*self.solver.settings.input_gain)
        self.readout_scale.fill_(1)

    def conductances(self, view=None):
        masters = tuple((MID_US + HALF_RANGE_US*u)/G_MAX_US for u in (self.u1,self.u2))
        if view is None:
            return masters
        # A fresh graph for each microbatch; the detached view is held across
        # the effective batch and all activation-checkpoint recomputation.
        return tuple(held + (m-m.detach()) for m,held in zip(masters,view))

    def analog(self, x, view=None):
        """Read-out MLP output before the digital fc2 bias."""
        g1,g2 = self.conductances(view)
        current = torch.cat((self.bias1,-self.bias1))*self.current_scale
        return self.solver(x,g1,g2,current,self.gains[0],self.gains[1]*self.readout_scale)

    def forward(self, x, view=None):
        return self.analog(x,view) + self.bias2


class OptDrn(nn.Module):
    def __init__(self, source, settings=DrnSettings()):
        super().__init__()
        self.start = source.start
        self.settings = settings
        self.digital = OptDecoder(source.config)
        self.digital.load_state_dict({k:v for k,v in source.state_dict().items() if k!='gains'})
        self.blocks = nn.ModuleList()
        for k,i in enumerate(range(self.start,len(source.layers))):
            layer = self.digital.layers[i]
            self.blocks.append(DrnMlp(layer.fc1,layer.fc2,source.gains[2*k:2*k+2],settings))
            layer.fc1 = layer.fc2 = None

    def forward(self,tokens,view=None,*,correction=None,capture=False):
        hidden = self.digital.embed(tokens)
        captures = []
        for i,layer in enumerate(self.digital.layers):
            if i < self.start:
                hidden = layer(hidden)
            else:
                hidden = layer.attention(hidden)
                k = i-self.start
                x = layer.final_layer_norm(hidden)
                y = self.blocks[k](x,None if view is None else view[k])
                if capture:
                    captures.append((x,y))
                if correction is not None:
                    a,b = correction[k]
                    y = y*a+b
                hidden = hidden + y
        out = self.digital.logits(hidden[:, :-1])
        return (out,captures) if capture else out

    @torch.no_grad()
    def analog_input(self,tokens):
        """Residual stream entering the first DRN decoder layer."""
        hidden = self.digital.embed(tokens)
        for layer in self.digital.layers[:self.start]:
            hidden = layer(hidden)
        return hidden

    def mlp_branch(self,k,hidden,view=None):
        """DRN MLP k split for output calibration: residual, analog output, digital bias.

        ``residual + (z + bias)`` is the operation order of ``forward``.
        """
        layer = self.digital.layers[self.start+k]
        hidden = layer.attention(hidden)
        z = self.blocks[k].analog(layer.final_layer_norm(hidden),None if view is None else view[k])
        return hidden, z, self.blocks[k].bias2

    def head(self,hidden):
        return self.digital.logits(hidden[:, :-1])

    def parameter_groups(self):
        masters = [u for b in self.blocks for u in (b.u1,b.u2)]
        calibration = [p for b in self.blocks for p in (b.bias1,b.bias2,b.gains)]
        calibration += [p for l in self.digital.layers[self.start:] for p in l.final_layer_norm.parameters()]
        used = {id(p) for p in masters+calibration}
        digital = [p for p in self.parameters() if id(p) not in used]
        return masters,calibration,digital


    @torch.no_grad()
    def clamp_masters(self):
        for u in self.parameter_groups()[0]:
            u.clamp_(-1,1)

    @torch.no_grad()
    def diagnostics(self,tokens,view=None):
        self.eval()
        hidden=self.digital.embed(tokens)
        rows=[]
        for i,layer in enumerate(self.digital.layers):
            if i<self.start:
                hidden=layer(hidden)
                continue
            hidden=layer.attention(hidden)
            x=layer.final_layer_norm(hidden)
            block=self.blocks[i-self.start]
            held=None if view is None else view[i-self.start]
            current=torch.cat((block.bias1,-block.bias1))*block.current_scale
            row=block.solver.diagnostics(x,*block.conductances(held),current,block.gains[0])
            rows.append({'layer':i,**row})
            hidden=hidden+block(x,held)
        return rows
