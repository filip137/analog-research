"""Deployment-time weight-to-conductance mappings of the OPT crossbar drift-regimes stage.

Shared record: on-chip-learning campaigns/drn-drift-mapping/series/002-crossbar (crossbar
mapping search, 17 November 2026). The frozen normalized targets ``q`` (per output channel,
|q| <= 1, a fixed digital output scale per channel) are unchanged; only the physical
conductances that hold them and the digital readout gain change. No retraining.

Each logical weight keeps two independently programmed and drifting devices (device 0, 1).
With the per-output-channel gain ``s`` and ``u = clip(s q, -1, 1)``:

- ``reference`` (the crossbar's own mapping): G0 = 48.6 + 39.6 u, G1 = 48.6 (a mid-window
  reference); q_read = (G0 - G1) / (39.6 s).
- ``push_pull``: G0 = 48.6 + 39.6 u, G1 = 48.6 - 39.6 u; q_read = (G0 - G1) / (79.2 s).
- ``two_sided``: with margin D, G0 = 9 + D + (79.2 - D) max(u, 0) and
  G1 = 9 + D + (79.2 - D) max(-u, 0); q_read = (G0 - G1) / ((79.2 - D) s).

Gain rules (``s`` per output channel of every analog matrix):

- ``unit``: s = 1.
- ``saturation`` with ``fraction`` p: s = 1 / |q|_(m+1), the (m+1)-th largest |q| of the
  channel, m = floor(p n); at most m of the channel's n weights saturate.
- ``fixed`` with ``value``: s = value.
- ``per_layer_absmax``: s = c / max(c |q|) over the matrix, with c the channel's digital
  scale, i.e. the matrix's largest effective weight maps to full scale (s <= 1).

``reference`` with the ``unit`` gain is the series-002 mapping, bit for bit. A common-mode
shift of both devices of a pair cancels in every mapping.
"""
import math

import torch

MID, HALF = 48.6, 39.6
PROGRAM_MIN, PROGRAM_MAX = 9.0, 88.2
KINDS = ('reference', 'push_pull', 'two_sided')
GAIN_KEYS = {'unit': {'rule'}, 'saturation': {'rule', 'fraction'}, 'fixed': {'rule', 'value'},
             'per_layer_absmax': {'rule'}}
BASE = {'kind': 'reference', 'gain': {'rule': 'unit'}, 'margin_us': 0.0}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid mapping {name}.')
    return float(value)


def parse(raw):
    """Canonical mapping settings: exactly kind, gain and margin_us."""
    if not isinstance(raw, dict) or set(raw) != set(BASE):
        raise ValueError(f'Expected exactly the mapping fields {sorted(BASE)}.')
    if raw['kind'] not in KINDS:
        raise ValueError(f'Expected a mapping kind in {KINDS}.')
    gain = raw['gain']
    if not isinstance(gain, dict) or gain.get('rule') not in GAIN_KEYS or set(gain) != GAIN_KEYS[gain['rule']]:
        raise ValueError('Expected exactly the fields of one mapping gain rule.')
    gain = dict(gain)
    if gain['rule'] == 'saturation':
        gain['fraction'] = _finite(gain['fraction'], 'saturation fraction')
        if not 0 < gain['fraction'] < 0.5:
            raise ValueError('Expected a saturation fraction in (0, 0.5).')
    if gain['rule'] == 'fixed':
        gain['value'] = _finite(gain['value'], 'gain value')
        if not 0 < gain['value'] <= 100:
            raise ValueError('Expected a fixed gain in (0, 100].')
    margin = _finite(raw['margin_us'], 'margin_us')
    if raw['kind'] == 'two_sided':
        if not 0 <= margin < PROGRAM_MAX - PROGRAM_MIN:
            raise ValueError('Expected a two-sided margin in [0, 79.2) uS.')
    elif margin != 0:
        raise ValueError('margin_us applies only to the two_sided mapping.')
    return {'kind': raw['kind'], 'gain': gain, 'margin_us': margin}


def channel_gains(gain, layout, q, scales):
    """Per-output-channel gains of every matrix (None for the unit rule)."""
    rule = gain['rule']
    if rule == 'unit':
        return None
    gains = []
    for k, spec in enumerate(layout):
        rows, cols = spec.shape
        a = q[spec.offset:spec.offset + spec.size].detach().reshape(rows, cols).abs().double()
        c = scales[k].double()
        if rule == 'fixed':
            s = torch.full((rows,), gain['value'], dtype=torch.float64, device=q.device)
        elif rule == 'saturation':
            m = int(math.floor(gain['fraction'] * cols))
            threshold = a.topk(m + 1, dim=1).values[:, m]
            s = torch.where(threshold > 0, 1 / threshold.clamp_min(1e-30), torch.ones_like(threshold))
        else:
            s = c / (c * a.amax(1)).max().clamp_min(1e-30)
        gains.append(s.float())
    return gains


def digital_scales(network):
    """Each matrix's per-output-channel digital scale (channel abs-max or the matrix scale)."""
    out = []
    for k, spec in enumerate(network.layout):
        if network.scaling == 'per_output_channel_absmax':
            out.append(network.channel_scales[network.channel_offsets[k]:network.channel_offsets[k + 1]])
        else:
            out.append(network.scales[k].expand(spec.shape[0]))
    return out


class Mapping:
    """One mapping bound to the network's layout and frozen targets."""

    def __init__(self, raw, network):
        self.spec = parse(raw)
        self.layout = list(network.layout[:network.base_matrices])
        if len(self.layout) != len(network.layout):
            raise ValueError('Deployment mappings support base matrices only (no analog adapters).')
        self.kind, self.margin = self.spec['kind'], self.spec['margin_us']
        self.gains = channel_gains(self.spec['gain'], self.layout, network.q, digital_scales(network))
        self.span = {'reference': HALF, 'push_pull': 2 * HALF,
                     'two_sided': PROGRAM_MAX - PROGRAM_MIN - self.margin}[self.kind]

    @property
    def is_base(self):
        return self.spec == BASE

    def _per_channel(self, flat, k, op):
        spec = self.layout[k]
        if self.gains is None:
            return flat
        s = self.gains[k].to(flat.device)[:, None]
        return op(flat.reshape(*flat.shape[:-1], *spec.shape), s).reshape(flat.shape)

    def scaled(self, q, k):
        """u = clip(s q, -1, 1) of matrix k (q itself under the unit gain)."""
        u = self._per_channel(q, k, lambda x, s: x * s)
        return u if self.gains is None else u.clamp(-1, 1)

    def targets(self, q):
        """(2, size) programmed conductance targets of every matrix, in layout order."""
        out = []
        for k, spec in enumerate(self.layout):
            qk = q[spec.offset:spec.offset + spec.size].detach()
            if self.kind == 'reference' and self.gains is None:
                out.append(torch.stack((MID + HALF * qk, torch.full_like(qk, MID))))      # series 002, bit for bit
                continue
            u = self.scaled(qk, k)
            if self.kind == 'reference':
                out.append(torch.stack((MID + HALF * u, torch.full_like(u, MID))))
            elif self.kind == 'push_pull':
                out.append(torch.stack((MID + HALF * u, MID - HALF * u)))
            else:
                low = PROGRAM_MIN + self.margin
                out.append(torch.stack((low + self.span * u.clamp_min(0), low + self.span * (-u).clamp_min(0))))
        return out

    def bind(self, physical):
        """Normalized weights read from (2, size) device conductances, in layout order."""
        parts = []
        for k, g in enumerate(physical):
            if self.kind == 'reference' and self.gains is None:
                parts.append((g[0] - g[1]) / HALF)                                         # series 002, bit for bit
                continue
            parts.append(self._per_channel((g[0] - g[1]) / self.span, k, lambda x, s: x / s))
        return torch.cat(parts)

    def clean(self, q):
        """The readout of exactly programmed targets: clip(s q, -1, 1) / s."""
        if self.gains is None:
            return q
        return torch.cat([self.clean_matrix(q[spec.offset:spec.offset + spec.size].detach(), k)
                          for k, spec in enumerate(self.layout)])

    def report(self, q, network):
        """Deterministic mapping statistics: gains, saturation and clean weight error."""
        scales = digital_scales(network)
        saturated = total = 0
        err = ref = 0.0
        gains = []
        for k, spec in enumerate(self.layout):
            qk = q[spec.offset:spec.offset + spec.size].detach()
            rows, cols = spec.shape
            c = scales[k].double()[:, None]
            if self.gains is not None:
                s = self.gains[k].double()
                saturated += int(((qk.reshape(rows, cols).double().abs() * s[:, None]) > 1).sum())
                gains.append(s)
            clean = self.clean_matrix(qk, k).reshape(rows, cols).double()
            w = qk.reshape(rows, cols).double()
            err += float(((clean - w) * c).square().sum())
            ref += float((w * c).square().sum())
            total += qk.numel()
        every = torch.cat(gains) if gains else None
        return {'mapping': dict(self.spec), 'span_us_per_unit_q': self.span,
                'saturated_fraction': saturated / total,
                'clean_relative_weight_error': math.sqrt(err / ref),
                'gain_quantiles': (None if every is None else
                                   {str(p): float(every.quantile(p)) for p in (0., .1, .5, .9, 1.)})}

    def clean_matrix(self, qk, k):
        if self.gains is None:
            return qk
        return self._per_channel(self.scaled(qk, k), k, lambda x, s: x / s)
