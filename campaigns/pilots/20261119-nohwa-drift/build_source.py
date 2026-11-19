"""Build the no-HWA model-scope CMO payload for the DRN from the FP32 teacher (exp-002, #33).

The analog part is the crossbar no-HWA source (hwa `pretrained` arm, method none): per-output-
channel q in [-1, 1] and channel scales, unit output gains, the teacher's MLP biases and norms.
That payload is analog-scope (no digital weights); this adds the teacher's digital weights as
``digital.*`` and declares scope 'model', which the DRN ``reconstruct_source`` requires.
Checks (all exact or reported): biases/norms equal the teacher's, q * scale against the
teacher's weights, digital key set equal to the HWA std3 payload convention.

    python build_source.py TEACHER_BIN CROSSBAR_PRETRAINED_PT ANALOG_LAYERS OUT.pt
"""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TEACHER_SHA256 = '2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9'
CROSSBAR = {6: '3442d7d2b4137c4126b2501a54d27a594b1ccbddc171b39d7329e2e4fd73a1b4',
            12: '480e1323aaed8a12036121d887f1aa38c3316b7cb39354ca14353cfc76c3abc8'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(teacher_path, crossbar_path, layers, out):
    from ebl.schema.opt_cmo_drn import Config
    from ebl.runtime.opt_cmo_drn.runtime import teacher_model
    from networks.opt.drn import reconstruct_source
    layers = int(layers)
    if digest(teacher_path) != TEACHER_SHA256 or digest(crossbar_path) != CROSSBAR[layers]:
        raise SystemExit('Unexpected teacher or crossbar no-HWA source.')
    teacher = teacher_model(teacher_path, Config(analog_layers=layers))
    crossbar = torch.load(crossbar_path, map_location='cpu', weights_only=True)
    state = teacher.state_dict()
    start = len(teacher.layers) - layers
    analog = [f'layers.{i}.{m}.' for i in range(start, len(teacher.layers)) for m in ('fc1', 'fc2', 'final_layer_norm')]
    model = {f'digital.{k}': v.clone() for k, v in state.items() if not any(k.startswith(a) for a in analog)}
    model.update({k: v.clone() for k, v in crossbar['model'].items()})
    checks = {'digital_keys': sum(k.startswith('digital.') for k in model)}
    q, scales = model['q'], model['channel_scales']
    offset = channel = 0
    worst = 0.0
    for k, i in enumerate(range(start, len(teacher.layers))):
        for j, name in enumerate(('fc1', 'fc2')):
            w = state[f'layers.{i}.{name}.weight']
            rebuilt = q[offset:offset + w.numel()].reshape_as(w) * scales[channel:channel + w.shape[0], None]
            worst = max(worst, float((rebuilt - w).abs().max() / w.abs().max()))
            if not torch.equal(model[f'mlp_biases.{2*k+j}'], state[f'layers.{i}.{name}.bias']):
                raise SystemExit('Crossbar MLP bias differs from the teacher.')
            offset += w.numel(); channel += w.shape[0]
        for p in ('weight', 'bias'):
            if not torch.equal(model[f'mlp_norms.{k}.{p}'], state[f'layers.{i}.final_layer_norm.{p}']):
                raise SystemExit('Crossbar MLP norm differs from the teacher.')
    if not bool((model['output_gains'] == 1).all()):
        raise SystemExit('Expected unit output gains.')
    checks.update(max_relative_weight_error=worst, analog_weights=offset)
    payload = {'provenance': {'schema': 'drn.nohwa_source.v1', 'teacher_sha256': TEACHER_SHA256,
                              'crossbar_pretrained_sha256': CROSSBAR[layers], 'analog_layers': layers,
                              'builder': 'campaigns/pilots/20261119-nohwa-drift/build_source.py'},
               'context': dict(crossbar['context']),
               'arm': {'id': 'pretrained', 'method': 'none', 'scope': 'model'},
               'model': model, 'fit': dict(crossbar['fit'])}
    source = reconstruct_source(teacher, payload, layers)  # raises on any layout mismatch
    tokens = torch.randint(0, 50000, (2, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = teacher(tokens).log_softmax(-1), source(tokens).log_softmax(-1)
    checks['source_vs_teacher_kl_random_tokens'] = float(torch.nn.functional.kl_div(b, a, log_target=True, reduction='batchmean'))
    torch.save(payload, out)
    checks['sha256'] = digest(out)
    print(json.dumps(checks))


if __name__ == '__main__':
    main(*sys.argv[1:5])
