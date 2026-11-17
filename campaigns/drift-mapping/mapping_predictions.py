"""Device-level predictions of the crossbar mapping search, frozen before any network run.

Shared record: on-chip-learning campaigns/drn-drift-mapping/series/002-crossbar (exp-xb1,
crossbar mapping search). No network forward is involved. For each candidate mapping
(training.xbar_mapping) on the series-002 std3 source (sha256 86522e23...), a random subset
of whole output channels of every analog matrix is programmed with the runner's own device
cells (training.drift_family.ReramDriftCells: CMO 2% endpoint, P&V N = 4 at 0.7431 verify
SDs, 60-s verify) under the declared law, then read at release + {0, 1 h, 24 h}.

Reported per mapping, law and age, pooled over the 12 matrices in effective-weight units
(W = q c, c the channel's fixed digital scale):

- saturated: fraction of weights with |s q| > 1 (whole network, deterministic);
- clean_error: RMS(W_clean - W) / RMS(W), the release-free cost of saturation;
- floor / ceiling: fraction of devices at the 5 uS / 150 uS drift bounds;
- slope: least-squares slope of the read on the target effective weight;
- rel_error: RMS(W_read - W) / RMS(W) against the frozen digital target W
  (saturation, programming error and drift together).

Usage: python campaigns/drift-mapping/mapping_predictions.py <source.pt> <lifecycle.json> <out.json>
                                                           [--rows N] [--multipliers 4 ...]
"""
import argparse
import copy
import hashlib
import json
import math
from types import SimpleNamespace
from pathlib import Path

import torch

from training import xbar_mapping as xm
from training.drift_family import ReramDriftCells, ReramFamily, program_verify

SHAPES = [(3072, 768), (768, 3072)] * 6
SOURCE_SHA256 = '86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028'


def spec(kind='reference', rule='unit', margin=0., **gain):
    return {'kind': kind, 'gain': {'rule': rule, **gain}, 'margin_us': margin}


CANDIDATES = {
    'base': spec(),
    'gain-sat0.2%': spec(rule='saturation', fraction=.002),
    'gain-sat1%': spec(rule='saturation', fraction=.01),
    'gain-x1.25': spec(rule='fixed', value=1.25),
    'gain-x1.5': spec(rule='fixed', value=1.5),
    'gain-x2': spec(rule='fixed', value=2.),
    'per-layer': spec(rule='per_layer_absmax'),
    'push-pull': spec('push_pull'),
    'two-sided': spec('two_sided'),
    'two-sided-m4': spec('two_sided', margin=4.),
    'two-sided-m8': spec('two_sided', margin=8.),
    'two-sided-m12': spec('two_sided', margin=12.),
    'two-sided-m16': spec('two_sided', margin=16.),
    # Combinations: span doubling with the small saturating gain.
    'push-pull+sat1%': spec('push_pull', 'saturation', fraction=.01),
    'two-sided+sat1%': spec('two_sided', 'saturation', fraction=.01),
}


def network(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Unexpected source checkpoint.')
    model = torch.load(path, map_location='cpu', weights_only=False)['model']
    layout, offset = [], 0
    for i, shape in enumerate(SHAPES):
        layout.append(SimpleNamespace(name=f"decoder.layers.{6 + i // 2}.{'fc1' if i % 2 == 0 else 'fc2'}",
                                      shape=shape, offset=offset, size=shape[0] * shape[1]))
        offset += shape[0] * shape[1]
    channel_offsets = [0]
    for shape in SHAPES:
        channel_offsets.append(channel_offsets[-1] + shape[0])
    q = model['q'].float()
    if q.numel() != offset or float(q.abs().max()) > 1:
        raise ValueError('Unexpected source targets.')
    return SimpleNamespace(layout=layout, base_matrices=len(layout), q=q, scaling='per_output_channel_absmax',
                           channel_scales=model['channel_scales'].float(), channel_offsets=channel_offsets,
                           scales=None), digest


def law_at(lifecycle, multiplier):
    raw = copy.deepcopy(json.loads(Path(lifecycle).read_text())['drift_regimes'])
    law = raw['law']
    base_m = float(raw['regime'].removeprefix('CMO-M'))
    for key in ('mean_us', 'spread_us', 'log_offset_us'):
        law[key] = law[key] / base_m * multiplier
    return ReramFamily(**law), raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('lifecycle')
    parser.add_argument('out')
    parser.add_argument('--rows', type=int, default=128, help='output channels sampled per matrix')
    parser.add_argument('--multipliers', type=float, nargs='+', default=[4.])
    parser.add_argument('--seed', type=int, default=20261117)
    args = parser.parse_args()
    torch.set_num_threads(8)
    net, digest = network(args.source)
    scales = xm.digital_scales(net)
    generator = torch.Generator().manual_seed(args.seed)
    picks = [torch.randperm(shape[0], generator=generator)[:args.rows].sort().values for shape in SHAPES]
    rows_out = []
    for label, raw in CANDIDATES.items():
        mapping = xm.Mapping(raw, net)
        report = mapping.report(net.q, net)
        targets = mapping.targets(net.q)
        for multiplier in args.multipliers:
            law, regimes = law_at(args.lifecycle, multiplier)
            acc = {}
            for k, spec_ in enumerate(net.layout):
                rows, cols = spec_.shape
                pick = picks[k]
                g = targets[k].reshape(2, rows, cols)[:, pick].reshape(2, -1).contiguous()
                w = (net.q[spec_.offset:spec_.offset + spec_.size].reshape(rows, cols)[pick]
                     * scales[k][pick, None]).double().flatten()
                s = torch.ones(len(pick)) if mapping.gains is None else mapping.gains[k][pick]
                denominator = (mapping.span * s.double() / scales[k][pick].double())[:, None]
                cells = ReramDriftCells(g.float(), law=law, seed=f"{regimes['seed']}/{regimes['assignment_seeds'][0]}",
                                        name=spec_.name)
                program_verify(cells, torch.arange(g.numel()), g.flatten(), attempts=regimes['deployment_attempts'],
                               threshold=regimes['deployment_threshold'], verify_seconds=regimes['verify_seconds'],
                               first_written=True)
                release = cells.clock
                clean = ((g[0] - g[1]).double().reshape(len(pick), cols) / denominator).flatten()
                for hours in (0., 1., 24.):
                    value = cells.advance(release + hours * 3600.)
                    read = ((value[0] - value[1]).double().reshape(len(pick), cols) / denominator).flatten()
                    a = acc.setdefault(hours, {'floor': 0, 'ceiling': 0, 'devices': 0, 'err': 0., 'ref': 0.,
                                               'xy': 0., 'xx': 0., 'drift': 0.})
                    a['floor'] += int((value <= law.drift_min).sum())
                    a['ceiling'] += int((value >= law.drift_max).sum())
                    a['devices'] += value.numel()
                    a['err'] += float((read - w).square().sum())
                    a['drift'] += float((read - clean).square().sum())
                    a['ref'] += float(w.square().sum())
                    a['xy'] += float((read * w).sum())
                    a['xx'] += float(w.square().sum())
            for hours, a in acc.items():
                rows_out.append({'mapping': label, 'spec': xm.parse(raw), 'multiplier': multiplier, 'hours': hours,
                                 'span_us': mapping.span, 'saturated': report['saturated_fraction'],
                                 'clean_error': report['clean_relative_weight_error'],
                                 'gain_quantiles': report['gain_quantiles'],
                                 'floor': a['floor'] / a['devices'], 'ceiling': a['ceiling'] / a['devices'],
                                 'slope': a['xy'] / a['xx'], 'rel_error': math.sqrt(a['err'] / a['ref']),
                                 'rel_error_vs_clean_mapped': math.sqrt(a['drift'] / a['ref'])})
                print(json.dumps({k: rows_out[-1][k] for k in ('mapping', 'multiplier', 'hours', 'saturated',
                                                               'floor', 'slope', 'rel_error')}), flush=True)
    meta = {'source_sha256': digest, 'lifecycle': str(args.lifecycle), 'rows_per_matrix': args.rows,
            'seed': args.seed, 'multipliers': args.multipliers,
            'sampled_weights': sum(args.rows * shape[1] for shape in SHAPES)}
    Path(args.out).write_text(json.dumps({'meta': meta, 'rows': rows_out}, indent=1) + '\n')


if __name__ == '__main__':
    main()
