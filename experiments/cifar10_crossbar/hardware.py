"""Matched, model-based IBM OM HWA, deployment and apparent-state recovery.

The reused plant owns hidden device parameters and persistent state. Recovery
writers receive only held apparent values and a pulse capability. Gradients and
Adam state are digital, as in the preceding MNIST recovery experiments.
"""
from dataclasses import asdict
from pathlib import Path
import os
import time

import torch
import torch.nn.functional as F

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.mnist_analog_relu.crossbar_common import (
    _sample_population, _program_endpoint, _matched_published_fault_overlay,
)
from training.ibm_om_standard_crossbar import (
    build_crossbar_layout, map_logical_weights, standard_crossbar_logits,
    IbmOmEffectiveCrossbarPlant, PulseAdam,
)
from training.closed_loop_crossbar_adam import ClosedLoopAdam
from .common import checkpoint, cpu_tree, evaluate, heartbeat, load_teacher, tensor_hash

SAMPLER = Path('/home/filip/miniconda3/envs/aihwkit/bin/python')
PROGRAM_TOLERANCE_Q = 0.04745
PROGRAM_MAXIMUM_PULSES = 128
PROGRAM_STREAM = 'cifar10_evaluation_programming_v1'


def validation_rank(metrics, epoch):
    return metrics['kl_teacher_student'], -metrics['accuracy'], epoch


def distillation_loss(logits, teacher_logits):
    logp = F.log_softmax(teacher_logits.detach(), dim=1)
    return (logp.exp() * (logp - F.log_softmax(logits, dim=1))).sum(dim=1).mean()


def sample(layout, assignment, policy, role, store):
    heartbeat(store, stage='sample_population', role=role, assignment=assignment)
    population, receipt, files = _sample_population(
        layout=layout, assignment_seed=assignment, corruption_policy=policy,
        sampler=Path(os.environ.get('EBL_AIHWKIT_PYTHON', str(SAMPLER))),
        artifact_root=store.run_dir / 'artifacts', role=role,
        bound_policy='native_sampled_bounds', required_aihwkit_version='1.1.0')
    atomic_write_json(store.run_dir / f'artifacts/{role}_population.json', receipt)
    return population, [path for path, _ in files]


def program(population, requested, endpoint, device):
    return _program_endpoint(
        population=population, requested=requested, assignment_seed=population.assignment_seed,
        endpoint_seed=endpoint, maximum_pulses=PROGRAM_MAXIMUM_PULSES,
        tolerance_x=PROGRAM_TOLERANCE_Q / 2, device=device,
        stream_role=PROGRAM_STREAM, random_stream_fingerprint=population.fingerprint)


def restore(population, state, device, *, published=None, healthy_p0=None):
    plant = IbmOmEffectiveCrossbarPlant(
        population, trajectory_seeds=state['trajectory_seeds'],
        trajectory_seed_derivation=state['trajectory_seed_derivation'], device=device)
    kwargs = {} if state['fault_transition'] is None else dict(
        expected_fault_source_population=published, expected_pre_fault_state=healthy_p0)
    plant.load_state_dict(state, **kwargs)
    if tensor_hash(plant.apparent) != tensor_hash(state['apparent']):
        raise RuntimeError('Restoring a checkpoint changed its held apparent state.')
    return plant


def snapshot(plant, writer, *, closed_loop):
    return dict(plant=plant.state_dict(), writer=writer.state_dict(),
                pulses=int(writer.pulse_count.sum()),
                verify_reads=int(writer.verify_reads) if closed_loop else 0,
                cells_at_pulse_cap=int((writer.pulse_count >= (
                    writer.total_pulse_cap if closed_loop else writer.pulse_cap_per_cell)).sum()))


def check_fault_immutability(plant):
    mask = plant.post_deployment_fault_mask
    if not torch.equal(plant.persistent[mask], plant.post_deployment_stuck_persistent_q[mask]):
        raise RuntimeError('A stuck physical cell moved during recovery.')


@torch.no_grad()
def prediction_changes(before, after, data):
    flips = fixed = broken = 0
    for x, y in data.batches('test'):
        p0, p1 = before(x).argmax(1), after(x).argmax(1)
        flips += int((p0 != p1).sum())
        fixed += int(((p0 != y) & (p1 == y)).sum())
        broken += int(((p0 == y) & (p1 != y)).sum())
    return dict(prediction_flips=flips, incorrect_to_correct=fixed, correct_to_incorrect=broken)


def run_hwa(request, config, data, device, store, teacher, teacher_hash, layout, initial, scales):
    s = config['settings']
    population, files = sample(layout, s['training_assignment'], 'counterfactual_repaired', 'training', store)
    development, extra = sample(layout, s['development_assignment'], 'counterfactual_repaired', 'development', store)
    files += extra
    minimum, maximum = population.logical_min.to(device), population.logical_max.to(device)
    master = initial.to(device).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([master], lr=config['learning_rate'], weight_decay=config['weight_decay'])
    noise_generator = torch.Generator(device=device).manual_seed(config['seed'])
    sigma = population.write_noise_std * population.nominal_dw_min * s['noise_multiplier']
    forward = lambda x, q: standard_crossbar_logits(x, q, layout, digital_scales=scales)
    selected, best, selected_payload = None, None, None
    start = time.time()
    for epoch in range(1, config['epochs'] + 1):
        loss_sum, examples = 0., 0
        for batch, (x, y) in enumerate(data.batches('train', epoch), 1):
            optimizer.zero_grad(set_to_none=True)
            bounded = torch.clamp(master, min=minimum, max=maximum)
            # Draw the same standard noise sequence in every multiplier arm,
            # including zero; the multiplicative intervention is the only change.
            noise = torch.randn(master.shape, generator=noise_generator, device=device) * sigma
            apparent = master + (bounded + noise - master).detach()
            loss = distillation_loss(forward(x, apparent), teacher.logits(x))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite HWA distillation loss.')
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                master.clamp_(-1, 1)
            loss_sum += float(loss.detach()) * len(y)
            examples += len(y)
            if batch % 100 == 0:
                heartbeat(store, stage='hwa', epoch=epoch, batch=batch)
        heartbeat(store, stage='hwa_development_programming', epoch=epoch)
        plant, programming = program(development, master.detach(), s['development_endpoint'], device)
        validation = evaluate(lambda x: forward(x, plant.apparent), data, 'validation', teacher)
        rank = validation_rank(validation, epoch)
        if best is None or rank < best:
            best = rank
            selected = dict(epoch=epoch, validation=validation, programming=programming)
            selected_payload = dict(master_q=master.detach().cpu().clone(), optimizer=cpu_tree(optimizer.state_dict()),
                                    noise_generator_state=noise_generator.get_state().cpu().clone())
        record = dict(stage='hwa', epoch=epoch, train_examples=examples, train_kl=loss_sum / examples,
                      validation=validation, selected_epoch=selected['epoch'], order_sha256=data.epoch_order_sha256,
                      elapsed_seconds=time.time() - start)
        store.append_metric(record)
        heartbeat(store, stage='hwa', epoch=epoch, selected_epoch=selected['epoch'])
        print(f"hwa noise={s['noise_multiplier']} epoch={epoch} val_acc={validation['accuracy']:.4f} KL={validation['kl_teacher_student']:.6f}", flush=True)
        del plant
    path = store.run_dir / 'checkpoints/hwa.pt'
    checkpoint(path, schema='cifar10.hwa_master.v1', teacher_sha256=teacher_hash, dims=config['dims'],
               data_receipt=data.receipt, digital_scales=scales, selected=selected,
               config=config, training_population_fingerprint=population.fingerprint,
               development_population_fingerprint=development.fingerprint, **selected_payload)
    return dict(selected=selected, teacher_sha256=teacher_hash, noise_multiplier=s['noise_multiplier'],
                sigma_q=sigma, checkpoint='checkpoints/hwa.pt',
                checkpoint_policy='minimum_healthy_development_validation_KL_epochs_1_through_budget',
                test_evaluated=False, evidence_class='model_based_aihwkit_preset'), files + [path]


def run_deploy(request, config, data, device, store, teacher, teacher_hash, layout, initial, scales):
    s = config['settings']
    requested, hwa_hash = initial.to(device), None
    if s['source'] == 'hwa':
        saved = torch.load(request.weights, map_location='cpu', weights_only=False)
        if (saved.get('schema') != 'cifar10.hwa_master.v1' or saved['teacher_sha256'] != teacher_hash
                or saved['data_receipt'] != data.receipt or tuple(saved['digital_scales']) != tuple(scales)):
            raise ValueError('HWA source does not match the frozen teacher, data and layer scales.')
        requested = saved['master_q'].to(device)
        hwa_hash = sha256_file(request.weights)
    healthy, files = sample(layout, s['assignment'], 'counterfactual_repaired', 'healthy', store)
    published, extra = sample(layout, s['assignment'], 'published', 'published', store)
    files += extra
    heartbeat(store, stage='deployment_programming')
    plant, programming = program(healthy, requested, s['endpoint'], device)
    healthy_state = plant.state_dict()
    forward = lambda x, q: standard_crossbar_logits(x, q, layout, digital_scales=scales)

    def scores(q):
        return {split: evaluate(lambda x: forward(x, q), data, split, teacher) for split in ('validation', 'test')}

    healthy_metrics = dict(apparent=scores(plant.apparent), persistent=scores(plant.persistent))
    mask, stuck, fault_report = _matched_published_fault_overlay(
        healthy=healthy, published=published, preset_default_corrupt_devices_prob=0.,
        enabled_corrupt_devices_prob=0.1348, corrupt_devices_range=0.01)
    transition = plant.apply_stuck_at_fault_transition(
        mask=mask, stuck_persistent_q=stuck, transition_id='cifar10_post_deployment_faults_v1',
        source_population_fingerprint=published.fingerprint)
    faulted_state = plant.state_dict()
    faulted_metrics = dict(apparent=scores(plant.apparent), persistent=scores(plant.persistent))
    bounded = requested.clamp(min=healthy.logical_min.to(device), max=healthy.logical_max.to(device))
    controls = dict(ideal_requested=scores(requested), support_clamped=scores(bounded))
    path = store.run_dir / 'checkpoints/deployment.pt'
    checkpoint(path, schema='cifar10.crossbar_deployment.v1', dims=config['dims'], teacher_sha256=teacher_hash,
               hwa_sha256=hwa_hash, data_receipt=data.receipt, digital_scales=scales, layout=[asdict(t) for t in layout],
               healthy_population=healthy.to('cpu'), published_population=published.to('cpu'),
               healthy_state=healthy_state, faulted_state=faulted_state, requested_q=requested,
               programming=programming, fault_report=fault_report, fault_transition=transition, config=config)
    metrics = dict(teacher_sha256=teacher_hash, hwa_sha256=hwa_hash, source=s['source'],
                   assignment=s['assignment'], endpoint=s['endpoint'], healthy=healthy_metrics,
                   faulted=faulted_metrics, controls=controls, programming=programming, faults=fault_report,
                   digital_scales=scales, cells=healthy.size, tiles=[asdict(t) for t in layout],
                   checkpoint='checkpoints/deployment.pt', evidence_class='model_based_aihwkit_preset')
    return metrics, files + [path]


def run_recover(request, config, data, device, store, teacher, teacher_hash, layout, initial, scales):
    s = config['settings']
    bundle = torch.load(request.device_state, map_location='cpu', weights_only=False)
    if (bundle.get('schema') != 'cifar10.crossbar_deployment.v1' or bundle['teacher_sha256'] != teacher_hash
            or bundle['data_receipt'] != data.receipt or tuple(bundle['dims']) != tuple(config['dims'])
            or tuple(bundle['digital_scales']) != tuple(scales) or bundle['config']['settings']['source'] != 'hwa'):
        raise ValueError('Recovery requires the matching, frozen HWA deployment bundle.')
    p0 = bundle[f"{s['condition']}_state"]
    restore_kwargs = dict(published=bundle['published_population'], healthy_p0=bundle['healthy_state'])
    plant = restore(bundle['healthy_population'], p0, device, **restore_kwargs)
    port = plant.local_star_update_port()
    closed_loop = s['writer'] == 'closed_loop_pv'
    if closed_loop:
        writer = ClosedLoopAdam(lambda: port.apparent, port.pulse, learning_rate=config['learning_rate'],
                                tolerance=s['verify_tolerance_q'], maximum_pulses=s['maximum_verify_pulses'],
                                total_pulse_cap=s['total_pulse_cap'])
    else:
        writer = PulseAdam(size=port.size, layout=layout, learning_rates=(config['learning_rate'],) * 2,
                           betas=(0.9, 0.999), epsilon=1e-8, layer_scope='all', nominal_dw_min=port.nominal_dw_min,
                           pulse_cap_per_cell=s['total_pulse_cap'],
                           generator=torch.Generator(device=device).manual_seed(config['seed']), device=device)
    forward = lambda x, q: standard_crossbar_logits(x, q, layout, digital_scales=scales)
    initial_validation = evaluate(lambda x: forward(x, plant.apparent), data, 'validation', teacher)
    selected = dict(epoch=0, validation=initial_validation)
    best, selected_state = validation_rank(initial_validation, 0), snapshot(plant, writer, closed_loop=closed_loop)
    history, start = [], time.time()
    for epoch in range(1, config['epochs'] + 1):
        writer.learning_rate_scale = 1. if s['schedule'] == 'constant' else 0.1 ** ((epoch - 1) / max(1, config['epochs'] - 1))
        loss_sum, examples, epoch_pulses, rounds = 0., 0, 0, 0
        for batch, (x, y) in enumerate(data.batches('train', epoch), 1):
            apparent = port.apparent.requires_grad_(True)
            loss = distillation_loss(forward(x, apparent), teacher.logits(x))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite recovery distillation loss.')
            gradient, = torch.autograd.grad(loss, apparent)
            outcome = writer.step(gradient) if closed_loop else writer.step(gradient, port)
            epoch_pulses += outcome['pulses'] if closed_loop else outcome['applied_pulses']
            rounds += outcome.get('verify_rounds', 0)
            loss_sum += float(loss.detach()) * len(y)
            examples += len(y)
            if batch % 100 == 0:
                heartbeat(store, stage='recover', epoch=epoch, batch=batch, pulses=epoch_pulses)
        check_fault_immutability(plant)
        validation = evaluate(lambda x: forward(x, plant.apparent), data, 'validation', teacher)
        rank = validation_rank(validation, epoch)
        if rank < best:
            best, selected = rank, dict(epoch=epoch, validation=validation)
            selected_state = snapshot(plant, writer, closed_loop=closed_loop)
        record = dict(stage='recover', epoch=epoch, train_examples=examples, train_kl=loss_sum / examples,
                      validation=validation, selected_epoch=selected['epoch'], order_sha256=data.epoch_order_sha256,
                      epoch_pulses=epoch_pulses, verify_rounds=rounds, elapsed_seconds=time.time() - start)
        history.append(record)
        store.append_metric(record)
        heartbeat(store, stage='recover', epoch=epoch, selected_epoch=selected['epoch'])
        print(f"recover {s['condition']} {s['writer']} epoch={epoch} val_acc={validation['accuracy']:.4f} KL={validation['kl_teacher_student']:.6f} pulses={epoch_pulses}", flush=True)
    final_state = snapshot(plant, writer, closed_loop=closed_loop)
    final_test = evaluate(lambda x: forward(x, plant.apparent), data, 'test', teacher)
    selected_plant = restore(bundle['healthy_population'], selected_state['plant'], device, **restore_kwargs)
    p0_apparent = p0['apparent'].to(device)
    p0_test = evaluate(lambda x: forward(x, p0_apparent), data, 'test', teacher)
    selected_test = evaluate(lambda x: forward(x, selected_plant.apparent), data, 'test', teacher)
    persistent_test = evaluate(lambda x: forward(x, selected_plant.persistent), data, 'test', teacher)
    changes = prediction_changes(lambda x: forward(x, p0_apparent), lambda x: forward(x, selected_plant.apparent), data)
    selected_cost = {k: selected_state[k] for k in ('pulses', 'verify_reads', 'cells_at_pulse_cap')}
    final_cost = {k: final_state[k] for k in ('pulses', 'verify_reads', 'cells_at_pulse_cap')}
    path = store.run_dir / 'checkpoints/recovery.pt'
    checkpoint(path, schema='cifar10.crossbar_recovery.v1', teacher_sha256=teacher_hash,
               deployment_sha256=sha256_file(request.device_state), config=config,
               selected=selected, selected_state=selected_state, final_state=final_state)
    return dict(teacher_sha256=teacher_hash, deployment_sha256=sha256_file(request.device_state),
                assignment=bundle['config']['settings']['assignment'], condition=s['condition'], writer=s['writer'],
                schedule=s['schedule'], initial_validation=initial_validation, initial_test=p0_test,
                selected=selected, selected_test=selected_test, selected_persistent_test=persistent_test,
                final_test=final_test, selected_cost=selected_cost, final_cost=final_cost, prediction_changes=changes,
                initial_apparent_sha256=tensor_hash(p0_apparent), initial_persistent_sha256=tensor_hash(p0['persistent']),
                selected_apparent_sha256=tensor_hash(selected_plant.apparent),
                checkpoint='checkpoints/recovery.pt', forward_state='held_apparent_q',
                checkpoint_policy='minimum_validation_KL_epoch_0_included',
                evidence_class='model_based_aihwkit_preset'), [path]


def run(request, config, data, device, store):
    teacher, teacher_hash = load_teacher(request.teacher_weights, config, data, device)
    layout = build_crossbar_layout(tuple(config['dims']), maximum_input_size=512)
    initial, scales = map_logical_weights(teacher.parameters(), layout, weight_scaling_omega=(1., 1.))
    runner = {'hwa': run_hwa, 'deploy': run_deploy, 'recover': run_recover}[config['stage']]
    return runner(request, config, data, device, store, teacher, teacher_hash, layout, initial, scales)
