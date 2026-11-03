"""ebl.runtime.drift_recovery — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from core.artifacts import sha256_file
from ebl.runtime.shared_recovery import run as run_recovery
from methods.maintenance.recovery import token_cohorts
import networks.opt as opt_mlp
from networks.opt_drift import CrossbarAdapter


def run(stage):
    from ebl.schema.source_fit import validate_source_fit
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
