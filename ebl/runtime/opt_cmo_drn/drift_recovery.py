"""Native custom writable-drift recovery on frozen four-cell DRN MLPs."""
import torch
import ebl.runtime.opt_cmo_drn.runtime as runtime
from networks.opt.data import pinned_load, load_corpus, TRANSFER_TESTS
from hardware.faults import state_hash
from ebl.runtime.shared_recovery import run as run_recovery
from methods.maintenance.recovery import token_cohorts


from networks.opt.adapter import DrnAdapter


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
