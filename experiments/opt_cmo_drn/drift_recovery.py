"""Native custom writable-drift recovery on frozen four-cell DRN MLPs."""
import torch
from experiments.opt_cmo_drn import runtime
from experiments.opt_cmo_drn.data import pinned_load, load_corpus, TRANSFER_TESTS
from experiments.opt_cmo_drn.faults import state_hash
from ebl.runtime.shared_recovery import run as run_recovery
from methods.maintenance.recovery import token_cohorts


class DrnAdapter:
    architecture = 'drn'

    def __init__(self, model, device):
        self.model = model.eval().requires_grad_(False)
        self.device = device

    def source_hash(self): return state_hash(self.model.state_dict())

    def clean(self, tokens): return self.model(tokens.to(self.device))

    def physical_layout(self):
        targets, groups, roles, names = [], [], [], []
        for k, block in enumerate(self.model.blocks):
            for j, u in enumerate((block.u1, block.u2)):
                target = (48.6+39.6*u.detach()).contiguous()
                row = torch.arange(u.shape[0], device=u.device)//(u.shape[0]//2)
                col = torch.arange(u.shape[1], device=u.device)
                col = col % 2 if j else col//(u.shape[1]//2)
                role = (4*j+row[:, None]*2+col[None, :]).to(torch.int8)
                targets.append(target)
                roles.append(role)
                groups.append(torch.full_like(role, k))
                names.append(f'decoder.layers.{self.model.start+k}.fc{j+1}')
        return targets, groups, roles, names

    def bind(self, physical):
        return [(physical[i]/88.2, physical[i+1]/88.2) for i in range(0, len(physical), 2)]

    def block(self, k, x, bound):
        return self.model.blocks[k](x, bound[k])

    def forward(self, tokens, physical=None, correction=None, capture=False):
        # The native solver uses extra truncated-BPTT sweeps only in training
        # mode. Frozen model parameters do not disable physical-view gradients.
        learning = torch.is_grad_enabled() and physical is not None and any(g.requires_grad for g in physical)
        for block in self.model.blocks: block.solver.train(learning)
        if physical is None and correction is None and not capture: return self.clean(tokens)
        bound = None if physical is None else self.bind(physical)
        digital = self.model.digital
        hidden = digital.embed(tokens.to(self.device))
        captures = []
        for i, layer in enumerate(digital.layers):
            if i < self.model.start:
                hidden = layer(hidden)
                continue
            hidden = layer.attention(hidden)
            k = i-self.model.start
            x = layer.final_layer_norm(hidden)
            y = self.model.blocks[k](x, None if bound is None else bound[k])
            if capture: captures.append((x, y))
            if correction is not None:
                a, b = correction[k]
                y = y*a+b
            hidden = hidden+y
        out = digital.logits(hidden[:, :-1])
        return (out, captures) if capture else out


def run(request):
    config, settings = request.spec, request.spec.drift_recovery
    if request.teacher_weights is None or request.weights is None:
        raise ValueError('Custom recovery needs pinned teacher and HWA weights.')
    if any(getattr(request, k, None) is not None for k in
           ('resume', 'device_state', 'base_weights', 'device_data', 'device_model', 'deployment')):
        raise ValueError('Unexpected custom-recovery input; event states replay internally.')
    runtime.configure_runtime(config)
    teacher = runtime.teacher_model(request.teacher_weights, config)
    source = pinned_load(request.weights, settings.source_sha256)
    model = runtime.from_mapped(teacher, config, source).to(config.device)
    del source, teacher
    streams = load_corpus(config.data.corpus, config.data.sha256)
    training, validation, identity = token_cohorts(streams, config.data.sequence_length, config.data.seed, settings)
    evaluations = {}
    # Preserve established held-out corpus identities; never use these for fits.
    for label, corpus, digest, count in TRANSFER_TESTS:
        if settings.canary: count = min(count, settings.validation_sequences)
        data = load_corpus(corpus, digest)['test']
        evaluations[f'transfer_{label}'] = data[:count*config.data.sequence_length].reshape(count, config.data.sequence_length)
    identity.update(corpus=config.data.corpus, corpus_sha256=config.data.sha256,
                    frozen_source_sha256=settings.source_sha256)
    store = runtime._store(request)
    try:
        result, artifacts = run_recovery(DrnAdapter(model, config.device), settings, training, validation,
                                        evaluations, identity, store, config.deployment.assignment_seed)
        store.complete(metrics=result, artifacts=[store.artifact_record(p, kind=k) for k, p in artifacts])
        return 0
    except BaseException as error:
        store.fail(error)
        raise
