"""Custom writable-drift stage for the existing OPT crossbar lifecycle."""
import torch
from torch.nn import functional as F

from experiments.artifacts import sha256_file
from experiments.cifar_crossbar.model import state_hash
from training.drift_recovery import run as run_recovery, token_cohorts
from workflow.networks import opt_mlp


class CrossbarAdapter:
    architecture = 'crossbar'

    def __init__(self, network, device):
        self.network = network.eval().requires_grad_(False)
        self.device = device

    def source_hash(self): return state_hash(self.network.state_dict())

    def clean(self, tokens): return self.network(tokens.to(self.device))

    def physical_layout(self):
        targets, groups, roles, names = [], [], [], []
        for k, matrix in enumerate(self.network.layout):
            q = self.network.q[matrix.offset:matrix.offset+matrix.size].detach()
            target = torch.stack((48.6+39.6*q, torch.full_like(q, 48.6)))
            role = torch.zeros_like(target, dtype=torch.int8)
            role[0] = 2*(k % 2)
            role[1] = 2*(k % 2)+1
            targets.append(target)
            roles.append(role)
            groups.append(torch.full_like(role, k//2))
            names.append(matrix.name)
        return targets, groups, roles, names

    def bind(self, physical):
        return torch.cat([(g[0]-g[1])/39.6 for g in physical])

    def block(self, k, x, bound):
        y = self.network._matrix(x, 2*k, bound)
        return self.network._matrix(F.relu(y), 2*k+1, bound)

    def forward(self, tokens, physical=None, correction=None, capture=False):
        if physical is None and correction is None and not capture: return self.clean(tokens)
        q = self.network.q if physical is None else self.bind(physical)
        digital = self.network.digital if self.network.digital is not None else self.network.teacher
        hidden = digital.embed(tokens.to(self.device))
        for i in range(self.network.start): hidden = digital.layers[i](hidden)
        captures = []
        for k, i in enumerate(range(self.network.start, len(digital.layers))):
            hidden = digital.layers[i].attention(hidden)
            x = self.network.mlp_norms[k](hidden)
            y = self.block(k, x, q)
            if capture: captures.append((x, y))
            if correction is not None:
                a, b = correction[k]
                y = y*a+b
            hidden = hidden+y
        out = digital.logits(hidden[:, :-1])
        return (out, captures) if capture else out


def run(stage):
    from workflow.runtime import validate_source_fit
    lc, settings = stage.lifecycle, stage.lifecycle.drift_recovery
    if sha256_file(stage.request.weights) != settings.source_sha256:
        raise ValueError('Custom recovery frozen source SHA-256 mismatch.')
    source = stage.checked('weights', 'hwa_source', hwa_arm=stage.selection.hwa_arm)
    validate_source_fit(lc, lc.hwa.arm(stage.selection.hwa_arm), source['fit'])
    stage.network.load_state_dict(source['model'])
    del source
    stage.cache()
    stage.device_bundle()
    corpus, _ = opt_mlp.load_corpus(lc.data, lc.network.model)
    training, validation, identity = token_cohorts(corpus['splits'], lc.data.sequence_length, lc.data.data_seed, settings)
    evaluations = {}
    for test in opt_mlp.transfer_tests(lc):
        payload, _ = opt_mlp.load_corpus(test, lc.network.model)
        count = min(test.sequences, settings.validation_sequences) if settings.canary else test.sequences
        evaluations[f'transfer_{test.label}'] = payload['splits'][test.split][:count*lc.data.sequence_length].reshape(count, lc.data.sequence_length)
    identity.update(corpus_sha256=lc.data.corpus_sha256, corpus=lc.data.corpus,
                    frozen_source_sha256=settings.source_sha256)
    return run_recovery(CrossbarAdapter(stage.network, stage.device), settings, training, validation,
                        evaluations, identity, stage.store, stage.selection.array_seed)
